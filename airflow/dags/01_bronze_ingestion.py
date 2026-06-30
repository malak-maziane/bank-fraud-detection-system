"""
DAG 01: Bronze Layer Ingestion
Loads CSV data from local storage to HDFS Bronze layer with partitioning by transactionDate.

Upstream: None (initial data source)
Downstream: 02_silver_processing.py
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
import logging

logger = logging.getLogger(__name__)

# DAG Configuration
default_args = {
    'owner': 'data-engineering',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
}

dag = DAG(
    dag_id='01_bronze_ingestion',
    default_args=default_args,
    description='Ingest bank transaction CSV to HDFS Bronze layer with date partitioning',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['bronze', 'ingestion', 'fraud-detection'],
)


def validate_csv_exists():
    """Validate that the source CSV file exists."""
    import os
    csv_path = '/project/bank_transactions_data_2.csv'
    
    if not os.path.exists(csv_path):
        raise AirflowException(f"CSV file not found: {csv_path}")
    
    file_size = os.path.getsize(csv_path) / (1024 * 1024)  # Convert to MB
    logger.info(f"✅ CSV file found: {csv_path} ({file_size:.2f} MB)")


def verify_hdfs_connection():
    """Verify HDFS is accessible via WebHDFS REST API."""
    import urllib.request
    import urllib.error

    url = "http://namenode:9870/webhdfs/v1/?op=LISTSTATUS"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            if resp.status == 200:
                logger.info("✅ HDFS connection verified via WebHDFS")
            else:
                raise AirflowException(f"HDFS WebHDFS returned status {resp.status}")
    except urllib.error.HTTPError as e:
        raise AirflowException(f"HDFS WebHDFS HTTP error: {e.code} {e.reason}")
    except Exception as e:
        raise AirflowException(f"Error verifying HDFS: {str(e)}")


def verify_bronze_data():
    """Verify Bronze data exists via WebHDFS."""
    import urllib.request
    import urllib.error

    url = "http://namenode:9870/webhdfs/v1/data/bronze/transactions?op=LISTSTATUS"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            if resp.status == 200:
                logger.info("✅ Bronze data verified via WebHDFS")
            else:
                raise AirflowException(f"Bronze path returned status {resp.status}")
    except urllib.error.HTTPError as e:
        raise AirflowException(f"Bronze data not found: {e.code} {e.reason}")
    except Exception as e:
        raise AirflowException(f"Error verifying Bronze data: {str(e)}")


# Tasks

task_validate_csv = PythonOperator(
    task_id='validate_csv_exists',
    python_callable=validate_csv_exists,
    doc='Validate that the source CSV file exists and is accessible',
    dag=dag,
)

task_verify_hdfs = PythonOperator(
    task_id='verify_hdfs_connection',
    python_callable=verify_hdfs_connection,
    doc='Verify HDFS cluster is accessible',
    dag=dag,
)

task_load_to_bronze = BashOperator(
    task_id='load_csv_to_hdfs_bronze',
    bash_command="""
    python /project/scripts/load_to_hdfs.py \
        /project/bank_transactions_data_2.csv \
        hdfs://namenode:9000/data/bronze/transactions
    """,
    doc='Load CSV to HDFS Bronze layer with partitioning by transactionDate',
    dag=dag,
)

task_verify_bronze = PythonOperator(
    task_id='verify_bronze_data',
    python_callable=verify_bronze_data,
    doc='Verify that data was successfully written to HDFS Bronze',
    dag=dag,
)

# Task Dependencies
task_validate_csv >> task_verify_hdfs >> task_load_to_bronze >> task_verify_bronze
