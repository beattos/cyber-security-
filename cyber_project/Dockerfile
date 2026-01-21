FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==2.1.3 \
    joblib==1.4.2 \
    scikit-learn==1.7.2

COPY . /app

CMD ["python", "-m", "pipeline.run_stream_demo", "--help"]
