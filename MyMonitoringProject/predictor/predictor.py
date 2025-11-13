import time
import schedule
from prometheus_api_client import PrometheusConnect
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


time.sleep(10)


PROMETHEUS_URL = "http://prometheus:9090"
PUSHGATEWAY_URL = "http://pushgateway:9091"
JOB_NAME = "disk_predictor_test"

max_retries = 5
for attempt in range(max_retries):
    try:
        prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)
        # Test connection
        prom.query("up")
        logger.info("Connected to Prometheus")
        break
    except Exception as e:
        logger.warning(f"Connection attempt {attempt+1}/{max_retries} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(5)
        else:
            logger.error("Failed to connect to Prometheus after retries")
            exit(1)

registry = CollectorRegistry()
g = Gauge(
    'disk_percentage_predicted_in_15_min',
    'Predicted disk usage % in next 15 minutes.',
    registry=registry
)


def predict_disk_usage():
    logger.info("\n Running short-term disk prediction test...")

    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        query = (
            '100 * (1 - (windows_logical_disk_free_bytes{job="windows-pc", volume="C:"} '
            '/ windows_logical_disk_size_bytes{job="windows-pc", volume="C:"}))'
        )

        result = prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step="1m"
        )

        if not result or len(result[0]['values']) < 5:
            logger.warning("Not enough recent data to train model.")
            g.set(0)  
            push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)
            return

        data = result[0]['values']
        df = pd.DataFrame(data, columns=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'], unit='s')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df['y'] = df['y'].interpolate(limit_direction='both')
        df = df.dropna()

        logger.info(f"Retrieved {len(df)} points from Prometheus for 1 hour.")

        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
        model.fit(df)

        future = model.make_future_dataframe(periods=15, freq='min')
        forecast = model.predict(future)

        predicted_value = forecast.iloc[-1]['yhat']
        logger.info(f"Predicted disk usage in 15 minutes: {predicted_value:.2f}%")

        g.set(predicted_value)
        push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)
        logger.info("Metric successfully pushed to Pushgateway.")

    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        g.set(0)
        push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)


schedule.every(2).minutes.do(predict_disk_usage)
predict_disk_usage()

while True:
    schedule.run_pending()
    time.sleep(5)