"""
DAG 02: Silver Layer Processing
Cleans, validates, and enriches data from Bronze layer with ML features.

Upstream: 01_bronze_ingestion
Downstream: 03_gold_aggregation, 04_ml_training
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import DagRun
from airflow.utils.state import State
from airflow.exceptions import AirflowException
import logging

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'data-engineering',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': True,
}

dag = DAG(
    dag_id='02_silver_processing',
    default_args=default_args,
    description='Clean, validate, and enrich Bronze data with ML features for Silver layer',
    schedule_interval='0 3 * * *',  # Daily at 3 AM (after Bronze)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['silver', 'processing', 'fraud-detection'],
)


def check_last_bronze_run_success():
    """Vérifie que la dernière exécution de 01_bronze_ingestion est en SUCCESS.
    Remplace l'ExternalTaskSensor : une seule requête SQL, pas d'attente/poking.
    """
    dag_runs = DagRun.find(dag_id='01_bronze_ingestion')
    if not dag_runs:
        raise AirflowException("Aucune exécution trouvée pour le DAG 01_bronze_ingestion")

    last_run = max(dag_runs, key=lambda dr: dr.execution_date)

    if last_run.state != State.SUCCESS:
        raise AirflowException(
            f"Dernière exécution de 01_bronze_ingestion en état '{last_run.state}' "
            f"(execution_date={last_run.execution_date}), pas SUCCESS"
        )

    logger.info(f"✅ Dernière exécution de 01_bronze_ingestion réussie ({last_run.execution_date})")


def verify_bronze_data():
    """Verify Bronze data exists before processing."""
    from pyspark.sql import SparkSession
    spark = None
    try:
        spark = SparkSession.builder \
            .appName("VerifyBronzeData") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .getOrCreate()

        hadoop_conf = spark._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
        path = spark._jvm.org.apache.hadoop.fs.Path("hdfs://namenode:9000/data/bronze/transactions")
        if fs.exists(path):
            logger.info("✅ Bronze data verified")
        else:
            raise AirflowException("Bronze data not found at hdfs://namenode:9000/data/bronze/transactions")
    except Exception as e:
        raise AirflowException(f"Error verifying Bronze data: {str(e)}")
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


def verify_silver_data():
    """Verify Silver data exists after processing."""
    from pyspark.sql import SparkSession
    spark = None
    try:
        spark = SparkSession.builder \
            .appName("VerifySilverData") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .getOrCreate()

        hadoop_conf = spark._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
        path = spark._jvm.org.apache.hadoop.fs.Path("hdfs://namenode:9000/data/silver/transactions_cleaned")
        if fs.exists(path):
            logger.info("✅ Silver data verified")
        else:
            raise AirflowException("Silver data not found at hdfs://namenode:9000/data/silver/transactions_cleaned")
    except Exception as e:
        raise AirflowException(f"Error verifying Silver data: {str(e)}")
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


# Tasks

task_check_bronze = PythonOperator(
    task_id='check_last_bronze_run_success',
    python_callable=check_last_bronze_run_success,
    doc='Vérifie que la dernière exécution de 01_bronze_ingestion est SUCCESS (rapide, sans polling)',
    dag=dag,
)

task_verify_bronze = PythonOperator(
    task_id='verify_bronze_data',
    python_callable=verify_bronze_data,
    doc='Verify that Bronze layer data exists',
    dag=dag,
)

task_process_silver = BashOperator(
    task_id='process_silver_layer',
    bash_command="""
    cd /project && \
    python /project/scripts/process_silver.py \
        hdfs://namenode:9000/data/bronze/transactions \
        hdfs://namenode:9000/data/silver/transactions_cleaned
    """,
    doc='Process Bronze → Silver: cleaning + ML features',
    dag=dag,
)

task_verify_silver = PythonOperator(
    task_id='verify_silver_data',
    python_callable=verify_silver_data,
    doc='Verify Silver layer data was created successfully',
    dag=dag,
)

# Task Dependencies
task_check_bronze >> task_verify_bronze >> task_process_silver >> task_verify_silver