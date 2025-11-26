FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

RUN mkdir -p outputs models models/prod_models

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV TF_CPP_MIN_LOG_LEVEL=2

COPY run_pipeline.sh /app/run_pipeline.sh
RUN chmod +x /app/run_pipeline.sh

CMD ["/app/run_pipeline.sh"]
