# Use a basic Python image
FROM python:3.9-slim

# Install the libraries we need
RUN pip install pandas prophet prometheus-api-client prometheus_client schedule

# Copy your script into the container
COPY predictor.py .

# Run the script when the container starts
CMD ["python", "predictor.py"]