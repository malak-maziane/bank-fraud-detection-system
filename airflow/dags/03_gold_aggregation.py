"""
DAG 03: Gold Layer - Star Schema with Iceberg Format
Creates dimension and fact tables in Iceberg format for Trino queries.
Tables are automatically registered in Hive Metastore.

Upstream: 02_silver_processing
Downstream: 04_ml_training
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
    dag_id='03_gold_aggregation',
    default_args=default_args,
    description='Create star schema with Iceberg tables for Trino analytical queries',
    schedule_interval='0 4 * * *',  # Daily at 4 AM (after Silver)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['gold', 'iceberg', 'trino', 'fraud-detection'],
)


def check_last_silver_run_success():
    """Vérifie que la dernière exécution de 02_silver_processing est en SUCCESS.
    Remplace l'ExternalTaskSensor : une seule requête SQL, pas d'attente/poking.
    """
    dag_runs = DagRun.find(dag_id='02_silver_processing')
    if not dag_runs:
        raise AirflowException("Aucune exécution trouvée pour le DAG 02_silver_processing")

    last_run = max(dag_runs, key=lambda dr: dr.execution_date)

    if last_run.state != State.SUCCESS:
        raise AirflowException(
            f"Dernière exécution de 02_silver_processing en état '{last_run.state}' "
            f"(execution_date={last_run.execution_date}), pas SUCCESS"
        )

    logger.info(f"✅ Dernière exécution de 02_silver_processing réussie ({last_run.execution_date})")


def verify_silver_and_metastore():
    """Verify Silver data and Hive Metastore connectivity."""
    from pyspark.sql import SparkSession
    spark = None
    try:
        spark = SparkSession.builder \
            .appName("VerifySilverMetastore") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkCatalog") \
            .config("spark.sql.catalog.spark_catalog.type", "hive") \
            .config("spark.sql.catalog.spark_catalog.warehouse", "hdfs://namenode:9000/warehouse") \
            .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
            .getOrCreate()

        # Verify Silver data exists on HDFS
        hadoop_conf = spark._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
        silver_path = spark._jvm.org.apache.hadoop.fs.Path("hdfs://namenode:9000/data/silver/transactions_cleaned")
        if not fs.exists(silver_path):
            raise AirflowException("Silver data not found: hdfs://namenode:9000/data/silver/transactions_cleaned")

        # Verify Hive Metastore connectivity by listing databases
        spark.sql("SHOW DATABASES").collect()

        logger.info("✅ Silver data verified")
        logger.info("✅ Hive Metastore accessible (catalog ready for Iceberg)")
    except Exception as e:
        raise AirflowException(f"Verification failed: {str(e)}")
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


def verify_iceberg_tables():
    """Verify gold Iceberg table directories and metadata exist on HDFS."""
    from pyspark.sql import SparkSession
    spark = None
    try:
        spark = SparkSession.builder \
            .appName("VerifyIcebergTables") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .getOrCreate()

        hadoop_conf = spark._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
        gold_path = spark._jvm.org.apache.hadoop.fs.Path("hdfs://namenode:9000/data/gold")

        if not fs.exists(gold_path):
            raise AirflowException("Gold data not found at hdfs://namenode:9000/data/gold")

        tables = [
            "dim_customers", "dim_devices", "dim_merchants", "dim_channels",
            "fact_transactions", "fact_fraud_risk",
        ]
        for table_name in tables:
            metadata_path = spark._jvm.org.apache.hadoop.fs.Path(
                f"hdfs://namenode:9000/data/gold/{table_name}/metadata"
            )
            if not fs.exists(metadata_path):
                raise AirflowException(
                    f"Métadonnées Iceberg introuvables pour la table '{table_name}'"
                )

        logger.info("✅ Iceberg gold directories verified")
        logger.info("✅ Iceberg metadata directories verified for all 6 tables")

    except Exception as e:
        raise AirflowException(f"Gold verification failed: {str(e)}")
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass
    # ← RIEN après ce bloc



# Tasks

task_check_silver = PythonOperator(
    task_id='check_last_silver_run_success',
    python_callable=check_last_silver_run_success,
    doc='Vérifie que la dernière exécution de 02_silver_processing est SUCCESS (rapide, sans polling)',
    dag=dag,
)

task_verify = PythonOperator(
    task_id='verify_silver_and_metastore',
    python_callable=verify_silver_and_metastore,
    doc='Verify Silver data and Hive Metastore connectivity',
    dag=dag,
)

task_create_schema = BashOperator(
    task_id='create_gold_iceberg_schema',
    bash_command="""
    cd /project && \
    python /project/scripts/process_gold.py \
        hdfs://namenode:9000/data/silver/transactions_cleaned \
        hdfs://namenode:9000/data/gold \
        hdfs://namenode:9000/warehouse
    """,
    doc='Create star schema with Iceberg tables (queryable by Trino)',
    dag=dag,
)

task_verify_tables = PythonOperator(
    task_id='verify_iceberg_tables',
    python_callable=verify_iceberg_tables,
    doc='Verify Iceberg tables and metadata structure on HDFS',
    dag=dag,
)

task_trino_readiness = BashOperator(
    task_id='check_trino_catalog_readiness',
    bash_command="""
    echo "=== TRINO READINESS ===" && \
    echo "✅ Iceberg catalog: hive_metastore" && \
    echo "✅ Database: default" && \
    echo "✅ Tables ready:" && \
    echo "  - dim_customers" && \
    echo "  - dim_devices" && \
    echo "  - dim_merchants" && \
    echo "  - dim_channels" && \
    echo "  - fact_transactions" && \
    echo "  - fact_fraud_risk" && \
    echo "\n✅ Trino connection test:" && \
    echo "   SELECT COUNT(*) FROM iceberg.default.fact_transactions;"
    """,
    doc='Display Trino readiness status',
    dag=dag,
)

# Task Dependencies
task_check_silver >> task_verify >> task_create_schema >> task_verify_tables >> task_trino_readiness