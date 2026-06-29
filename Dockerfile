FROM python:3.11-slim

WORKDIR /app

COPY requirements-base.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements-base.txt

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
