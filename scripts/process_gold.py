#!/usr/bin/env python3
"""
Gold Layer Processing: Star Schema design with Iceberg format for Trino.
Creates dimension tables and fact tables for analytical queries.

Iceberg enables:
- Trino SQL queries via Hive Metastore catalog
- ACID transactions
- Schema evolution
- Time travel
"""

import sys
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, row_number, dense_rank, max as spark_max, min as spark_min,
    current_timestamp
)
from pyspark.sql.window import Window
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_gold_schema_iceberg(
    silver_path: str = "hdfs://namenode:9000/data/silver/transactions_cleaned",
    gold_path: str = "hdfs://namenode:9000/data/gold",
    warehouse_path: str = "hdfs://namenode:9000/warehouse"
) -> None:
    """
    Create Gold layer with star schema using Iceberg format.
    Tables are registered with Hive Metastore and queryable via Trino.
    
    Args:
        silver_path: HDFS path to Silver data
        gold_path: HDFS path for Gold output
        warehouse_path: HDFS warehouse location for Iceberg
    """
    try:
        # ============================================
        # SPARK SESSION WITH ICEBERG SUPPORT
        # ============================================
        spark = SparkSession.builder \
            .appName("GoldLayerIcebergSchema") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkCatalog") \
            .config("spark.sql.catalog.spark_catalog.type", "hive") \
            .config("spark.sql.catalog.spark_catalog.warehouse", warehouse_path) \
            .config("spark.sql.catalog.spark_catalog.hadoop.iceberg.engine.hive.lock-enabled", "false") \
            .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
            .getOrCreate()
        
        logger.info(f"✅ Spark Session initialized with Iceberg support")
        
        # ============================================
        # READ SILVER DATA
        # ============================================
        logger.info(f"Reading Silver data from: {silver_path}")
        
        df_silver = spark.read.parquet(silver_path)
        initial_rows = df_silver.count()
        logger.info(f"✅ Silver data loaded: {initial_rows} rows")
        
        # ============================================
        # DIMENSION TABLES
        # ============================================
        logger.info("Creating Dimension Tables...")
        
        # DIM_CUSTOMERS: Customer profile dimension
        dim_customers = df_silver.select(
            col("AccountID").alias("customer_id"),
            col("CustomerAge").alias("age"),
            col("CustomerOccupation").alias("occupation"),
            col("Channel").alias("preferred_channel"),
            col("AccountBalance").alias("account_balance"),
            when(col("IsAbnormalAge") == 1, "High Risk").otherwise("Normal").alias("age_category"),
            spark_max(col("TransactionAmount")).over(
                Window.partitionBy("AccountID")
            ).alias("max_transaction_amount"),
            current_timestamp().alias("dim_load_date")
        ).dropDuplicates(["customer_id"])
        
        write_iceberg_table(spark, dim_customers, f"{gold_path}/dim_customers", "dim_customers")
        
        # DIM_DEVICES: Device/Hardware dimension
        dim_devices = df_silver.select(
            col("DeviceID").alias("device_id"),
            col("DeviceIDPrefix").alias("device_prefix"),
            col("IPFirstOctet").alias("ip_first_octet"),
            col("IP Address").alias("ip_address"),
            current_timestamp().alias("dim_load_date")
        ).dropDuplicates(["device_id"])
        
        write_iceberg_table(spark, dim_devices, f"{gold_path}/dim_devices", "dim_devices")
        
        # DIM_MERCHANTS: Merchant dimension
        dim_merchants = df_silver.select(
            col("MerchantID").alias("merchant_id"),
            col("Location").alias("merchant_location"),
            current_timestamp().alias("dim_load_date")
        ).dropDuplicates(["merchant_id"])
        
        write_iceberg_table(spark, dim_merchants, f"{gold_path}/dim_merchants", "dim_merchants")
        
        # DIM_CHANNELS: Transaction channel dimension
        dim_channels = df_silver.select(
            col("Channel").alias("channel_id")
        ).distinct().withColumn(
            "channel_description",
            when(col("channel_id") == "ATM", "Automated Teller Machine")
            .when(col("channel_id") == "Online", "Online Banking")
            .when(col("channel_id") == "Mobile", "Mobile Application")
            .otherwise("Other")
        ).withColumn(
            "risk_level",
            when(col("channel_id").isin(["ATM", "Online"]), "High")
            .otherwise("Medium")
        ).withColumn("dim_load_date", current_timestamp())
        
        write_iceberg_table(spark, dim_channels, f"{gold_path}/dim_channels", "dim_channels")
        
        # ============================================
        # FACT TABLES
        # ============================================
        logger.info("Creating Fact Tables...")
        
        # FACT_TRANSACTIONS: Main transactional fact table
        fact_transactions = df_silver.select(
            col("TransactionID").alias("transaction_id"),
            col("AccountID").alias("customer_id"),
            col("MerchantID").alias("merchant_id"),
            col("DeviceID").alias("device_id"),
            col("Channel").alias("channel_id"),
            col("TransactionAmount").alias("amount"),
            col("TransactionType").alias("transaction_type"),
            col("TransactionDateTime").alias("transaction_timestamp"),
            col("LoginAttempts").alias("login_attempts"),
            col("TransactionDuration").alias("duration_seconds"),
            col("IsHighRiskLocation").alias("is_high_risk_location"),
            col("IsHighRiskChannel").alias("is_high_risk_channel"),
            col("AmountToBalanceRatio").alias("amount_to_balance_ratio"),
            col("DaysSincePreviousTransaction").alias("days_since_previous"),
            current_timestamp().alias("fact_load_date")
        )
        
        write_iceberg_table(spark, fact_transactions, f"{gold_path}/fact_transactions", "fact_transactions")
        
        # FACT_FRAUD_RISK: Risk assessment fact table
        fact_fraud_risk = df_silver.select(
            col("TransactionID").alias("transaction_id"),
            col("AccountID").alias("customer_id"),
            (
                col("IsFrequentLogins").cast("int") +
                col("IsHighRiskLocation").cast("int") +
                col("IsHighRiskChannel").cast("int") +
                col("IsLargeTransaction").cast("int")
            ).alias("fraud_risk_score"),
            when(col("LoginAttempts") > 3, 1).otherwise(0).alias("suspicious_login_flag"),
            when(col("AmountToBalanceRatio") > 0.5, 1).otherwise(0).alias("unusual_amount_flag"),
            col("TransactionDateTime").alias("risk_assessment_timestamp"),
            current_timestamp().alias("fact_load_date")
        )
        
        write_iceberg_table(spark, fact_fraud_risk, f"{gold_path}/fact_fraud_risk", "fact_fraud_risk")
        
        # ============================================
        # SUMMARY & VERIFICATION
        # ============================================
        logger.info("\n=== GOLD LAYER SUMMARY ===")
        logger.info(f"✅ Dimension Tables (4): dim_customers, dim_devices, dim_merchants, dim_channels")
        logger.info(f"✅ Fact Tables (2): fact_transactions, fact_fraud_risk")
        logger.info(f"✅ Format: Apache Iceberg (queryable by Trino via Hive Metastore)")
        logger.info(f"✅ Total records processed: {initial_rows}")
        logger.info(f"✅ Output path: {gold_path}")
        
        spark.stop()
        
    except Exception as e:
        logger.error(f"❌ Error in Gold processing: {str(e)}")
        raise


def write_iceberg_table(spark, df, path: str, table_name: str) -> None:
    """
    Write DataFrame to Iceberg table format.
    Tables are automatically registered in Hive Metastore.
    
    Args:
        spark: SparkSession
        df: DataFrame to write
        path: HDFS path for table
        table_name: Table name in Metastore
    """
    try:
        # Count rows before write
        row_count = df.count()
        
        # Write as Iceberg table
        df.write \
            .format("iceberg") \
            .mode("overwrite") \
            .option("path", path) \
            .saveAsTable(table_name, path=path)
        
        logger.info(f"✅ Table '{table_name}' written ({row_count} rows) → {path}")
        
    except Exception as e:
        logger.error(f"❌ Error writing Iceberg table '{table_name}': {str(e)}")
        raise


if __name__ == "__main__":
    silver_path = "hdfs://namenode:9000/data/silver/transactions_cleaned"
    gold_path = "hdfs://namenode:9000/data/gold"
    warehouse_path = "hdfs://namenode:9000/warehouse"
    
    if len(sys.argv) > 1:
        silver_path = sys.argv[1]
    if len(sys.argv) > 2:
        gold_path = sys.argv[2]
    if len(sys.argv) > 3:
        warehouse_path = sys.argv[3]
    
    logger.info("Starting Gold Layer Processing with Iceberg...")
    create_gold_schema_iceberg(silver_path, gold_path, warehouse_path)
    logger.info("Gold processing complete! ✅")
