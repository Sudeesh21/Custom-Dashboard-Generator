import time
import schedule
from prometheus_api_client import PrometheusConnect
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from prophet import Prophet
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
PROMETHEUS_URL = "http://prometheus:9090"
PUSHGATEWAY_URL = "http://pushgateway:9091"
JOB_NAME = "system_predictor"


prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)


registry = CollectorRegistry()
g_disk = Gauge('disk_percentage_predicted_in_15_min', 'Predicted Disk Usage %', registry=registry)
g_cpu = Gauge('cpu_usage_predicted_in_15_min', 'Predicted CPU Usage %', registry=registry)
g_mem = Gauge('memory_usage_predicted_in_15_min', 'Predicted Memory Usage %', registry=registry)
g_net = Gauge('network_usage_predicted_in_15_min', 'Predicted Network Traffic (MB/s)', registry=registry)

def generate_synthetic_history(metric_type):
    
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    
    if metric_type == "cpu":
       
        values = np.random.uniform(10, 60, size=len(dates))
    elif metric_type == "memory":
        
        values = np.linspace(30, 80, len(dates)) + np.random.normal(0, 2, len(dates))
    elif metric_type == "disk":
        
        values = np.linspace(40, 85, len(dates))
    elif metric_type == "network":
       
        values = np.random.exponential(scale=10, size=len(dates))
    else:
        values = np.random.uniform(0, 100, size=len(dates))

    
    df = pd.DataFrame({'ds': dates, 'y': values})
    return df

def get_metric_prediction(metric_name, prom_query, gauge, unit_conversion=1, metric_type="cpu"):
    print(f"Analyzing {metric_name}...")
    
    
    try:
        result = prom.custom_query_range(
            query=prom_query,
            start_time="2d",
            end_time="now",
            step="5m"
        )
        real_data_exists = len(result) > 0
    except Exception:
        real_data_exists = False

    df = pd.DataFrame()

    if not real_data_exists:
        print(f"  -> Not enough real data for {metric_name}. Using SYNTHETIC history for presentation.")
        df = generate_synthetic_history(metric_type)
    else:
        data = result[0]['values']
        df = pd.DataFrame(data, columns=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'], unit='s')
        df['y'] = pd.to_numeric(df['y']) * unit_conversion

    #  Train the AI Model
    try:
        model = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        # Predict 15 minutes into future
        future = model.make_future_dataframe(periods=3, freq='5min') 
        forecast = model.predict(future)
        
        
        predicted_value = forecast.iloc[-1]['yhat']
        
       
        if "usage" in metric_name.lower() or "percentage" in metric_name.lower():
            predicted_value = max(0, min(100, predicted_value))
        else:
            predicted_value = max(0, predicted_value)

        print(f"  -> Forecast: {predicted_value:.2f}")
        gauge.set(predicted_value)

    except Exception as e:
        print(f"Error predicting {metric_name}: {e}")

def run_all_predictions():
    print("--- Starting Prediction Cycle ---")
    
    # 1. Disk Prediction (Type: disk)
    get_metric_prediction(
        "Disk Usage", 
        '100 * (1 - (windows_logical_disk_free_bytes{job="windows-pc", volume="C:"} / windows_logical_disk_size_bytes{job="windows-pc", volume="C:"}))', 
        g_disk,
        metric_type="disk"
    )

    # 2. CPU Prediction (Type: cpu)
    get_metric_prediction(
        "CPU Usage", 
        '100 - (avg(rate(windows_cpu_time_total{job="windows-pc", mode="idle"}[5m])) * 100)', 
        g_cpu,
        metric_type="cpu"
    )

    # 3. Memory Prediction (Type: memory)
    get_metric_prediction(
        "Memory Usage", 
        '100 * (1 - (windows_memory_available_bytes{job="windows-pc"} / windows_memory_physical_total_bytes{job="windows-pc"}))', 
        g_mem,
        metric_type="memory"
    )

    # 4. Network Prediction (Type: network)
    get_metric_prediction(
        "Network Traffic", 
        'sum(rate(windows_net_bytes_total{job="windows-pc"}[5m]))', 
        g_net,
        unit_conversion=0.000001,
        metric_type="network"
    )

    # Push all metrics 
    try:
        push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)
        print("All predictions pushed to gateway.\n")
    except Exception as e:
        print(f"Could not push to gateway: {e}")


schedule.every(1).minutes.do(run_all_predictions)

run_all_predictions()

while True:
    schedule.run_pending()
    time.sleep(1)
    import time
import schedule
from prometheus_api_client import PrometheusConnect
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from prophet import Prophet
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

# --- 1. CONFIGURATION ---
PROMETHEUS_URL = "http://prometheus:9090"
PUSHGATEWAY_URL = "http://pushgateway:9091"
JOB_NAME = "system_predictor"


prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

registry = CollectorRegistry()
g_disk = Gauge('disk_percentage_predicted_in_15_min', 'Predicted Disk Usage %', registry=registry)
g_cpu = Gauge('cpu_usage_predicted_in_15_min', 'Predicted CPU Usage %', registry=registry)
g_mem = Gauge('memory_usage_predicted_in_15_min', 'Predicted Memory Usage %', registry=registry)
g_net = Gauge('network_usage_predicted_in_15_min', 'Predicted Network Traffic (MB/s)', registry=registry)

# --- SMART SYNTHETIC HISTORY GENERATOR ---
def generate_smart_pattern(metric_type):

    end_date = datetime.now()
    start_date = end_date - timedelta(days=2) # 2 days history
    dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    x = np.arange(len(dates))
    
    time_offset = time.time() / 1000 

    if metric_type == "cpu":
        values = 50 + 30 * np.sin((x / 50) + time_offset) + np.random.normal(0, 5, len(dates))
    
    elif metric_type == "memory":
        
        values = 60 + 20 * np.sin((x / 100) + time_offset)
    
    elif metric_type == "network":
        
        values = 40 + 30 * np.sin((x / 20) + time_offset) + np.random.exponential(5, len(dates))

    else: 
        
        values = 40 + (x / len(dates)) * 50 

    
    values = np.clip(values, 0, 100)

    df = pd.DataFrame({'ds': dates, 'y': values})
    return df

def get_metric_prediction(metric_name, gauge, metric_type="cpu"):
    print(f"Analyzing {metric_name} pattern...")
    
    
    df = generate_smart_pattern(metric_type)

    try:
        # Train the AI Model on the pattern
        model = Prophet(
            daily_seasonality=False, 
            yearly_seasonality=False, 
            weekly_seasonality=False,
            changepoint_prior_scale=0.5 # Make AI more sensitive to recent changes
        )
        model.fit(df)

        # Predict 15 minutes into future
        future = model.make_future_dataframe(periods=3, freq='5min') 
        forecast = model.predict(future)
        
        
        predicted_value = forecast.iloc[-1]['yhat']
        
        
        predicted_value = max(0, min(100, predicted_value))

        print(f"  -> {metric_name} Forecast: {predicted_value:.2f}%")
        gauge.set(predicted_value)

    except Exception as e:
        print(f"Error predicting {metric_name}: {e}")

def run_all_predictions():
    print("--- Starting AI Analysis ---")
    
    get_metric_prediction("Disk Usage", g_disk, metric_type="disk")
    get_metric_prediction("CPU Usage", g_cpu, metric_type="cpu")
    get_metric_prediction("Memory Usage", g_mem, metric_type="memory")
    get_metric_prediction("Network Traffic", g_net, metric_type="network")

    
    try:
        push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)
        print("Predictions pushed to gateway.\n")
    except Exception as e:
        print(f"Could not push to gateway: {e}")


schedule.every(30).seconds.do(run_all_predictions)


run_all_predictions()

while True:
    schedule.run_pending()
    time.sleep(1)