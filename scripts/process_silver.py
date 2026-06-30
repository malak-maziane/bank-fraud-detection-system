#!/usr/bin/env python3
"""
Silver Layer Processing: Data cleaning, validation, and feature engineering.
Reads from Bronze (Parquet), cleans data, calculates ML features, writes to Silver.
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, datediff, dayofweek, month, hour, isnan, 
    isnull, count, avg, stddev, substring, split, to_date
)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_silver_layer(
    bronze_path: str = "hdfs://namenode:9000/data/bronze/transactions",
    silver_path: str = "hdfs://namenode:9000/data/silver/transactions_cleaned"
) -> None:
    """
    Process Bronze data to Silver layer with cleaning and feature engineering.
    
    Args:
        bronze_path: HDFS path to Bronze data
        silver_path: HDFS path for Silver output
    """
    try:
        # Initialize Spark Session
        spark = SparkSession.builder \
            .appName("SilverLayerProcessing") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .getOrCreate()
        
        logger.info(f"Reading Bronze data from: {bronze_path}")
        
        # Read Bronze data
        df_bronze = spark.read.parquet(bronze_path)
        
        initial_count = df_bronze.count()
        logger.info(f"✅ Bronze data loaded. Initial rows: {initial_count}")
        
        # ============================================
        # STEP 1: DATA VALIDATION & CLEANING
        # ============================================
        logger.info("Starting data cleaning...")
        
        # Remove duplicates
        df_clean = df_bronze.dropDuplicates(["TransactionID"])
        
        # Remove rows with null values in critical columns
        critical_cols = ["TransactionID", "AccountID", "TransactionAmount", "TransactionDate"]
        df_clean = df_clean.dropna(subset=critical_cols)
        
        # Remove negative amounts (data quality issue)
        df_clean = df_clean.filter(col("TransactionAmount") > 0)
        
        # Remove invalid transaction types
        valid_types = ["Debit", "Credit", "Transfer", "Withdrawal"]
        df_clean = df_clean.filter(col("TransactionType").isin(valid_types))
        
        # Remove rows with invalid IP addresses
        df_clean = df_clean.filter(
            (col("IP Address") != "") & 
            (col("IP Address").isNotNull())
        )
        
        rows_after_cleaning = df_clean.count()
        rows_removed = initial_count - rows_after_cleaning
        logger.info(f"Rows removed during cleaning: {rows_removed}")
        logger.info(f"Rows after cleaning: {rows_after_cleaning}")
        
        # ============================================
        # STEP 2: FEATURE ENGINEERING FOR ML
        # ============================================
        logger.info("Calculating ML features...")
        
        # Convert string timestamps to proper format
        df_features = df_clean.withColumn(
            "TransactionDateTime",
            col("TransactionDate").cast(TimestampType())
        )
        
        # Time-based features
        df_features = df_features \
            .withColumn("TransactionHour", hour(col("TransactionDateTime"))) \
            .withColumn("TransactionDayOfWeek", dayofweek(col("TransactionDateTime"))) \
            .withColumn("TransactionMonth", month(col("TransactionDateTime"))) \
            .withColumn(
                "DaysSincePreviousTransaction",
                datediff(col("TransactionDateTime"), col("PreviousTransactionDate"))
            )
        
        # Amount-based features
        df_features = df_features \
            .withColumn("AmountToBalanceRatio", 
                       col("TransactionAmount") / (col("AccountBalance") + 1)) \
            .withColumn("IsLargeTransaction",
                       when(col("TransactionAmount") > 5000, 1).otherwise(0)) \
            .withColumn("IsSmallTransaction",
                       when(col("TransactionAmount") < 10, 1).otherwise(0))
        
        # Device/Location features
        df_features = df_features \
            .withColumn("DeviceIDPrefix", substring(col("DeviceID"), 1, 4)) \
            .withColumn("IPFirstOctet", split(col("IP Address"), "\\.").getItem(0)) \
            .withColumn("IsHighRiskLocation",
                       when(col("Location").isin(["Unknown", ""]), 1).otherwise(0))
        
        # Account behavior features
        df_features = df_features \
            .withColumn("IsFrequentLogins",
                       when(col("LoginAttempts") > 3, 1).otherwise(0)) \
            .withColumn("IsAbnormalAge",
                       when((col("CustomerAge") < 18) | (col("CustomerAge") > 100), 1).otherwise(0))
        
        # Channel-based features
        df_features = df_features \
            .withColumn("IsHighRiskChannel",
                       when(col("Channel").isin(["ATM", "Online"]), 1).otherwise(0))
        
        logger.info(f"✅ ML features calculated. Total columns: {len(df_features.columns)}")
        logger.info(f"New features created: {len(df_features.columns) - len(df_clean.columns)}")
        
        # ============================================
        # STEP 3: WRITE TO SILVER LAYER
        # ============================================
        logger.info(f"Writing to Silver layer: {silver_path}")
        
        # Partition par jour (et non par timestamp complet) pour éviter de créer
        # une partition par transaction, ce qui ralentirait fortement l'écriture.
        df_features = df_features.withColumn(
            "TransactionDateOnly", to_date(col("TransactionDateTime"))
        )
        
        df_features.write \
            .mode("overwrite") \
            .partitionBy("TransactionDateOnly") \
            .parquet(silver_path)
        
        logger.info(f"✅ Silver data written successfully!")
        logger.info(f"Output path: {silver_path}")
        
        # Statistics
        logger.info(f"\n=== PROCESSING SUMMARY ===")
        logger.info(f"Initial rows (Bronze): {initial_count}")
        logger.info(f"Final rows (Silver): {rows_after_cleaning}")
        logger.info(f"Rows removed: {rows_removed} ({100*rows_removed/initial_count:.1f}%)")
        logger.info(f"Output format: Parquet (partitioned by TransactionDateOnly)")
        logger.info(f"ML features: {len(df_features.columns) - len(df_clean.columns)} new features")
        
        spark.stop()
        
    except Exception as e:
        logger.error(f"❌ Error in Silver processing: {str(e)}")
        raise


if __name__ == "__main__":
    bronze_path = "hdfs://namenode:9000/data/bronze/transactions"
    silver_path = "hdfs://namenode:9000/data/silver/transactions_cleaned"
    
    if len(sys.argv) > 1:
        bronze_path = sys.argv[1]
    if len(sys.argv) > 2:
        silver_path = sys.argv[2]
    
    logger.info("Starting Silver Layer Processing...")
    process_silver_layer(bronze_path, silver_path)
    logger.info("Silver processing complete! ✅")