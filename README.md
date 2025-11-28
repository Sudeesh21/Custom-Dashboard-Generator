# Windows System Monitoring with AI Predictive Alerting
A comprehensive, containerized solution for monitoring Windows environments. This project goes beyond traditional real-time metrics by integrating a custom Artificial Intelligence agent that forecasts future system resource usage, allowing administrators to proactively address issues before they cause downtime.

 # Key Features & Novelty
What makes this project unique is the seamless integration of host-level Windows metrics with containerized analysis tools and proactive AI forecasting.

 **AI-Powered Predictive Maintenance (Novelty)**: Unlike standard monitoring that alerts after a threshold is crossed, this system includes a custom Python AI predictor (using Facebook Prophet). It analyzes historical data to forecast future trends, generating alerts like "Disk will be full in 15 minutes" or "CPU saturation predicted soon."

**Deep Windows Integration:** Utilizes the official windows_exporter running natively on the host to capture deep system metrics (CPU cores, memory pages, logical disks, network I/O, services, and processes).

**Full-Stack Containerization:** Prometheus, Grafana, Pushgateway, and the AI Predictor run in isolated Docker containers orchestrated by Docker Compose, ensuring consistency and portability.

**Automated "One-Click" Deployment (Novelty):** A sophisticated setup script handles the complex task of installing the Windows service on the host machine and launching the containerized environment in a single step.

**Rich Visualization & Alerting:** Comes pre-provisioned with a detailed Grafana dashboard and separate alerting rules for both real-time system status and AI predictions.

# Architecture Overview

**1. Windows Exporter:** Runs as a service on the host Windows PC, exposing system metrics.

**2. Prometheus:** Scrapes metrics from the Windows Exporter and the Pushgateway.

**3. AI Predictor (Python):** Queries historical data from Prometheus, performs time-series forecasting, and pushes predictions to the Pushgateway.

**4. Grafana:** Visualizes data from Prometheus and manages alerting rules.
