FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY memory_poc ./memory_poc
RUN pip install --no-cache-dir --no-deps .
COPY data ./data
COPY tests ./tests
COPY scripts ./scripts
ENV PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
ENTRYPOINT ["memory-poc"]
