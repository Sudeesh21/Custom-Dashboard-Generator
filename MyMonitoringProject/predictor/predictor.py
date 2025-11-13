import os
import time
import schedule
import logging
from prometheus_api_client import PrometheusConnect
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
JOB_NAME = os.getenv("PREDICTOR_JOB", "disk_predictor_test")

# Wait for services to be ready
time.sleep(10)

# Connect to Prometheus
prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

registry = CollectorRegistry()
g = Gauge('disk_percentage_predicted_in_15_min',
          'Predicted disk usage % in next 15 minutes.',
          registry=registry)

def predict_disk_usage():
    logger.info("Running disk prediction job")
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)

        query = (
            '100 * (1 - (windows_logical_disk_free_bytes{job="windows-pc", volume="C:"} '
            '/ windows_logical_disk_size_bytes{job="windows-pc", volume="C:"}))'
        )
        result = prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step="60s"
        )

        if not result or len(result[0].get('values', [])) < 5:
            logger.warning("Not enough data points — setting metric to 0")
            g.set(0)
            push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)
            return

        data = result[0]['values']
        df = pd.DataFrame(data, columns=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'], unit='s')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.interpolate(limit_direction='both').dropna()

        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
        model.fit(df)

        future = model.make_future_dataframe(periods=15, freq='min')
        forecast = model.predict(future)
        predicted_value = forecast.iloc[-1]['yhat']
        predicted_value = max(0, min(100, predicted_value))  # Clamp between 0-100

        logger.info(f"Predicted disk usage in 15 minutes: {predicted_value:.2f}%")
        g.set(predicted_value)
        push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        g.set(0)
        push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)

schedule.every(2).minutes.do(predict_disk_usage)

while True:
    schedule.run_pending()
    time.sleep(5)
