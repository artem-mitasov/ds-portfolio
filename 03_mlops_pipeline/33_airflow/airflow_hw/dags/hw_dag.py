import datetime as dt
import os
import sys

from airflow.models import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ---- Настройка проекта ----
# Пусть проект может быть перенесён куда угодно
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PROJECT_PATH'] = project_path
sys.path.insert(0, project_path)

# ---- Импорты модулей ----
from modules.pipeline import pipeline
from modules.predict import predict

# ---- Аргументы DAG ----
default_args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

# ---- DAG ----
with DAG(
    dag_id='car_price_prediction',
    schedule='0 15 * * *',  # через cron, пример: каждый день в 15:00
    default_args=default_args,
    catchup=False,
) as dag:

    # ---- Задачи ----
    pipeline_task = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
    )

    predict_task = PythonOperator(
        task_id='predict',
        python_callable=predict,
    )

    # ---- Порядок выполнения ----
    pipeline_task >> predict_task