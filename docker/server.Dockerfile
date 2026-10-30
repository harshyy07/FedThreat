FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir numpy pydantic fastapi "uvicorn[standard]" cryptography && pip install --no-cache-dir --no-deps -e .
RUN useradd -m app && mkdir /certs && chown app /certs
USER app
# --ssl-cert-reqs 2 == ssl.CERT_REQUIRED: clients WITHOUT a CA-signed certificate cannot connect.
CMD ["uvicorn", "fedthreat.server.app:app", "--host", "0.0.0.0", "--port", "8443", \
     "--ssl-keyfile", "/certs/server.key", "--ssl-certfile", "/certs/server.crt", \
     "--ssl-ca-certs", "/certs/ca.crt", "--ssl-cert-reqs", "2"]
