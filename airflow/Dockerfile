FROM apache/airflow:2.6.3-python3.8

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Installer Java (nécessaire pour Spark) + curl (pour télécharger le JAR Iceberg)
# -- doit rester en root pour apt-get
RUN apt-get -o Acquire::Retries=3 update \
    && apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        openjdk-11-jre-headless curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# IMPORTANT : repasser en utilisateur airflow AVANT les installations pip.
# Sinon pip installe dans /root/.local, invisible pour le worker qui tourne
# en tant qu'utilisateur "airflow" (HOME=/home/airflow) au runtime.
USER airflow

# IMPORTANT : pyspark==3.3.4 utilise un setup.py "legacy" (egg_info) qui est
# incompatible avec les versions récentes de setuptools (>=69), qui ont changé
# la signature interne de canonicalize_version(). On épingle donc une version
# de setuptools connue pour fonctionner avec ce type de package.
RUN python -m pip install --upgrade pip "setuptools==68.2.2" wheel
RUN python -m pip install --no-cache-dir --default-timeout=300 --retries 5 --prefer-binary \
    pyspark==3.3.4 \
    pandas \
    numpy \
    scikit-learn \
    pyarrow \
    joblib \
    cloudpickle \
    apache-airflow-providers-apache-hive \
    apache-airflow-providers-apache-spark

# Vérification : pip show ne suffit pas, il faut un vrai import pour confirmer
# que le package est accessible par l'utilisateur airflow au runtime.
RUN python -m pip show pyspark
RUN python -c "import pyspark; print('✅ pyspark importable, version:', pyspark.__version__)"

# IMPORTANT : runtime Iceberg (JAR Java/Scala), indispensable pour que Spark
# résolve la classe org.apache.iceberg.spark.SparkCatalog. pip install des
# bindings Python d'Iceberg ne suffit pas, il faut le JAR JVM dans le
# classpath de pyspark (dossier jars/ du package, pas un module pip).
# Version 1.4.3 compatible avec pyspark==3.3.4 (Spark 3.3.x / Scala 2.12).
# -- exécuté en USER airflow pour que le fichier appartienne au bon user.
ARG ICEBERG_VERSION=1.4.3
RUN PYSPARK_JARS_DIR=$(python -c "import pyspark, os; print(os.path.join(os.path.dirname(pyspark.__file__), 'jars'))") && \
    curl -fsSL -o "${PYSPARK_JARS_DIR}/iceberg-spark-runtime-3.3_2.12-${ICEBERG_VERSION}.jar" \
        "https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.3_2.12/${ICEBERG_VERSION}/iceberg-spark-runtime-3.3_2.12-${ICEBERG_VERSION}.jar"

# Vérification : confirme que le JAR est bien présent et lisible par l'user airflow
RUN python -c "import pyspark, os, glob; jars=os.path.join(os.path.dirname(pyspark.__file__),'jars'); assert any('iceberg-spark-runtime' in f for f in glob.glob(jars+'/*.jar')), 'JAR Iceberg manquant'; print('✅ JAR Iceberg présent')"