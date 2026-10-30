FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir numpy pydantic httpx cryptography fastapi && pip install --no-cache-dir --no-deps -e .
RUN useradd -m app
USER app
CMD ["python", "scripts/run_client.py"]
