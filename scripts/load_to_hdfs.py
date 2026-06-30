#!/usr/bin/env python3
"""
Load bank transactions CSV to HDFS Bronze layer with partitioning by transactionDate.
Partitions are created as: transactionDate=YYYY-MM-DD/data.parquet
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_csv_to_hdfs_bronze(
    csv_path: str,
    hdfs_output_path: str = "hdfs://namenode:9000/data/bronze/transactions",
    partition_col: str = "TransactionDate"
) -> None:
    """
    Load CSV file to HDFS Bronze layer with partitioning.
    
    Args:
        csv_path: Local path to CSV file
        hdfs_output_path: HDFS destination path
        partition_col: Column to partition by (will extract date part only)
    
    Returns:
        None
    """
    try:
        # Initialize Spark Session
        spark = SparkSession.builder \
            .appName("LoadToHDFSBronze") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-common:3.2.1") \
            .config("spark.sql.sources.commitProtocolClass", "org.apache.spark.sql.execution.datasources.SQLHadoopMapReduceCommitProtocol") \
            .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2") \
            .getOrCreate()
        
        logger.info(f"Reading CSV file: {csv_path}")
        
        # Le defaultFS est hdfs://, donc tout chemin sans schéma est interprété
        # comme un chemin HDFS. On préfixe explicitement avec file:// pour lire
        # depuis le système de fichiers local du conteneur.
        local_csv_path = f"file://{csv_path}" if not csv_path.startswith(("file://", "hdfs://")) else csv_path
        logger.info(f"Resolved local path for Spark: {local_csv_path}")
        
        # Read CSV
        df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(local_csv_path)
        
        logger.info(f"CSV loaded. Rows: {df.count()}, Columns: {len(df.columns)}")
        logger.info(f"Columns: {', '.join(df.columns)}")
        
        # Extract date part from TransactionDate for partitioning
        # Format: "2023-04-11 16:29:14" -> partition: "2023-04-11"
        df_with_partition = df.withColumn(
            "transactionDate",
            to_date(col("TransactionDate"), "yyyy-MM-dd HH:mm:ss")
        )
        
        # Write to HDFS with partitioning
        logger.info(f"Writing to HDFS: {hdfs_output_path}")
        logger.info("Partitioning by: transactionDate")
        
        df_with_partition.write \
            .mode("overwrite") \
            .partitionBy("transactionDate") \
            .parquet(hdfs_output_path)
        
        logger.info(f"✅ Data successfully written to HDFS Bronze layer")
        logger.info(f"Output path: {hdfs_output_path}")
        
        # Show statistics
        partitions = df_with_partition.select("transactionDate").distinct().count()
        logger.info(f"Number of partitions created: {partitions}")
        
        spark.stop()
        
    except Exception as e:
        logger.error(f"❌ Error loading CSV to HDFS: {str(e)}")
        raise


if __name__ == "__main__":
    # Default paths
    csv_path = "/project/bank_transactions_data_2.csv"
    hdfs_output_path = "hdfs://namenode:9000/data/bronze/transactions"
    
    # Allow override via command line arguments
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        hdfs_output_path = sys.argv[2]
    
    logger.info(f"Starting Bronze Layer ingestion...")
    logger.info(f"CSV input: {csv_path}")
    logger.info(f"HDFS output: {hdfs_output_path}")
    
    load_csv_to_hdfs_bronze(csv_path, hdfs_output_path)
    
    logger.info("Ingestion complete! ✅")
