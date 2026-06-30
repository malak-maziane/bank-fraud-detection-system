#!/usr/bin/env python3
"""
ML Model Training: Fraud Detection using RandomForest.
Reads features from Silver layer, trains model, and saves for production inference.
"""

import sys
import os
import pickle
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_fraud_detection_model(
    silver_path: str = "hdfs://namenode:9000/data/silver/transactions_cleaned",
    model_output_dir: str = "/project/ml/models"
) -> None:
    """
    Train RandomForest fraud detection model on Silver layer data.

    Args:
        silver_path: HDFS path to Silver data with features
        model_output_dir: Local directory to save model
    """
    spark = None
    try:
        # ============================================
        # STEP 1: INITIALIZE SPARK & READ DATA
        # ============================================
        logger.info("Initializing Spark Session...")

        spark = SparkSession.builder \
            .appName("FraudDetectionMLTraining") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .getOrCreate()

        logger.info(f"Reading Silver data from: {silver_path}")

        df = spark.read.parquet(silver_path)
        total_rows = df.count()
        logger.info(f"✅ Data loaded: {total_rows} rows")

        # ============================================
        # STEP 2: PREPARE FEATURES FOR ML
        # ============================================
        logger.info("Preparing features for training...")

        numerical_features = [
            "TransactionAmount",
            "LoginAttempts",
            "TransactionDuration",
            "AccountBalance",
            "CustomerAge",
            "TransactionHour",
            "TransactionDayOfWeek",
            "TransactionMonth",
            "DaysSincePreviousTransaction",
            "AmountToBalanceRatio",
            "IsLargeTransaction",
            "IsSmallTransaction",
            "IsHighRiskLocation",
            "IsFrequentLogins",
            "IsAbnormalAge",
            "IsHighRiskChannel",
        ]

        df_ml = df.select(numerical_features)
        pdf = df_ml.toPandas()

        logger.info(f"Features shape: {pdf.shape}")
        logger.info(f"Features: {list(pdf.columns)}")

        # Handle missing values
        pdf = pdf.fillna(pdf.mean(numeric_only=True))

        # ============================================
        # FRAUD LABEL : score basé sur OR pondéré
        # Fraud = au moins 2 facteurs de risque présents
        # ============================================
        pdf["fraud_risk_score"] = (
            pdf["IsFrequentLogins"].astype(int) +
            pdf["IsHighRiskLocation"].astype(int) +
            pdf["IsHighRiskChannel"].astype(int) +
            pdf["IsLargeTransaction"].astype(int)
        )
        pdf["fraud_label"] = (pdf["fraud_risk_score"] >= 2).astype(int)

        X = pdf[numerical_features]
        y = pdf["fraud_label"]

        fraud_count = y.sum()
        legitimate_count = (y == 0).sum()
        logger.info(f"Target distribution:")
        logger.info(f"  - Legitimate: {legitimate_count} ({100 * legitimate_count / len(y):.1f}%)")
        logger.info(f"  - Fraud:      {fraud_count} ({100 * fraud_count / len(y):.1f}%)")

        if fraud_count == 0:
            raise ValueError(
                "Aucune transaction frauduleuse détectée après labellisation. "
                "Vérifier les colonnes IsFrequentLogins / IsHighRiskLocation / "
                "IsHighRiskChannel / IsLargeTransaction dans les données Silver."
            )

        # ============================================
        # STEP 3: TRAIN/TEST SPLIT
        # ============================================
        logger.info("Splitting data (80% train / 20% test)...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set:     {len(X_test)} samples")

        # ============================================
        # STEP 4: TRAIN RANDOMFOREST MODEL
        # ============================================
        logger.info("Training RandomForest Classifier...")
        logger.info("  - n_estimators=100")
        logger.info("  - max_depth=15")
        logger.info("  - min_samples_split=10")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        model.fit(X_train, y_train)
        logger.info("✅ Model training complete!")

        # ============================================
        # STEP 5: EVALUATE MODEL
        # ============================================
        logger.info("\n=== MODEL EVALUATION ===")

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy:     {test_accuracy:.4f}")
        logger.info(f"Test Precision:    {test_precision:.4f}")
        logger.info(f"Test Recall:       {test_recall:.4f}")
        logger.info(f"Test F1-Score:     {test_f1:.4f}")

        # AUC-ROC : uniquement si les deux classes sont présentes dans y_test
        n_classes_test = len(np.unique(y_test))
        test_auc = None
        if n_classes_test >= 2:
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_pred_proba_test)
            logger.info(f"Test AUC-ROC:      {test_auc:.4f}")
        else:
            logger.warning("⚠️ AUC-ROC non calculable : une seule classe dans y_test")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"  TN: {tn}  FP: {fp}")
            logger.info(f"  FN: {fn}  TP: {tp}")
        else:
            logger.warning(f"⚠️ Confusion matrix inattendue : shape={cm.shape}")

        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': numerical_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"\nTop 10 Feature Importance:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # ============================================
        # STEP 6: SAVE MODEL
        # ============================================
        os.makedirs(model_output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_output_dir, f"fraud_detector_{timestamp}.pkl")

        logger.info(f"\nSaving model to: {model_path}")

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info("✅ Model saved successfully!")

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'model_type': 'RandomForestClassifier',
            'n_features': len(numerical_features),
            'features': numerical_features,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'fraud_rate': float(y.mean()),
            'total_samples': len(pdf),
            'training_date': datetime.now().isoformat()
        }

        metadata_path = os.path.join(model_output_dir, f"metadata_{timestamp}.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Metadata saved to: {metadata_path}")

        # Symlink vers le modèle le plus récent
        latest_link = os.path.join(model_output_dir, "fraud_detector_latest.pkl")
        if os.path.exists(latest_link) or os.path.islink(latest_link):
            os.remove(latest_link)
        os.symlink(model_path, latest_link)
        logger.info(f"Latest model link: {latest_link}")

        logger.info("\n=== TRAINING COMPLETE ===")
        logger.info("Model ready for production inference! 🚀")

    except Exception as e:
        logger.error(f"❌ Error during ML training: {str(e)}")
        raise
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


if __name__ == "__main__":
    silver_path = "hdfs://namenode:9000/data/silver/transactions_cleaned"
    model_output_dir = "/project/ml/models"

    if len(sys.argv) > 1:
        silver_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_output_dir = sys.argv[2]

    logger.info("Starting Fraud Detection ML Training Pipeline...")
    train_fraud_detection_model(silver_path, model_output_dir)