FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.128.8 "uvicorn[standard]==0.39.0" httpx==0.28.1
COPY relay.py .
ENV PORT=8080
CMD ["sh", "-c", "uvicorn relay:app --host 0.0.0.0 --port ${PORT}"]
