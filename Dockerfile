FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app
RUN useradd --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown -R app:app /app /data

USER app
EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["python", "-m", "raglab.cli", "serve", "--host", "0.0.0.0", "--port", "8000", "--db", "/data/rag.db"]
