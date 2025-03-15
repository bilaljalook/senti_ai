FROM python:3.10.6-slim

WORKDIR /app

COPY requirements.txt .
COPY setup.py .

RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt && \
    pip install .

COPY senti_ai/ senti_ai/

CMD uvicorn senti_ai.api.fast:app --host 0.0.0.0 --port 5367
