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
    'owner': 'data-science',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'depends_on_past': False,
    'email': ['datascience@example.com'],
    'email_on_failure': True,
}

dag = DAG(
    dag_id='04_ml_training',
    default_args=default_args,
    description='Train fraud detection ML model on Silver data',
    schedule_interval='0 5 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'fraud-detection'],
)


def check_last_silver_run_success():
    dag_runs = DagRun.find(dag_id='02_silver_processing')
    if not dag_runs:
        raise AirflowException("Aucune exécution trouvée pour 02_silver_processing")

    last_run = max(dag_runs, key=lambda dr: dr.execution_date)

    if last_run.state != State.SUCCESS:
        raise AirflowException(
            f"Dernière exécution de 02_silver_processing en état '{last_run.state}' "
            f"(execution_date={last_run.execution_date}), pas SUCCESS"
        )

    logger.info(f"✅ Dernière exécution de 02_silver_processing réussie ({last_run.execution_date})")


def verify_silver_ml_features():
    from pyspark.sql import SparkSession
    spark = None
    try:
        spark = SparkSession.builder \
            .appName("VerifySilverMLFeatures") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .getOrCreate()

        hadoop_conf = spark._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
        silver_path = spark._jvm.org.apache.hadoop.fs.Path(
            "hdfs://namenode:9000/data/silver/transactions_cleaned"
        )
        if not fs.exists(silver_path):
            raise AirflowException(
                "Silver data not found at hdfs://namenode:9000/data/silver/transactions_cleaned"
            )
        logger.info("✅ Silver data with ML features verified")

    except Exception as e:
        raise AirflowException(f"Error verifying Silver data: {str(e)}")
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


# Tasks

task_check_silver = PythonOperator(
    task_id='check_last_silver_run_success',
    python_callable=check_last_silver_run_success,
    doc='Vérifie que la dernière exécution de 02_silver_processing est SUCCESS (sans polling)',
    dag=dag,
)

task_verify_silver = PythonOperator(
    task_id='verify_silver_ml_features',
    python_callable=verify_silver_ml_features,
    doc='Verify Silver layer contains ML features',
    dag=dag,
)

task_train_model = BashOperator(
    task_id='train_fraud_detection_model',
    bash_command="""
    cd /project && \
    python /project/scripts/train_model.py \
        hdfs://namenode:9000/data/silver/transactions_cleaned \
        /project/ml/models
    """,
    dag=dag,
)

task_evaluate_model = BashOperator(
    task_id='evaluate_model_performance',
    bash_command="""
    if [ -f /project/ml/models/fraud_detector_latest.pkl ]; then
        echo "✅ Model evaluation complete" && \
        ls -lh /project/ml/models/
    else
        echo "❌ Model file not found!" && exit 1
    fi
    """,
    dag=dag,
)

task_validate_model = BashOperator(
    task_id='validate_model_for_production',
    bash_command="""
    echo "=== PRODUCTION VALIDATION ===" && \
    if [ -f /project/ml/models/fraud_detector_latest.pkl ]; then
        python3 << 'EOF'
import pickle, os
model_path = "/project/ml/models/fraud_detector_latest.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f"✅ Model type: {type(model).__name__}")
print(f"✅ n_features: {model.n_features_in_}")
print(f"✅ classes: {list(model.classes_)}")
EOF
    else
        echo "❌ Model not found" && exit 1
    fi
    """,
    dag=dag,
)

# Task Dependencies
task_check_silver >> task_verify_silver >> task_train_model >> task_evaluate_model >> task_validate_model