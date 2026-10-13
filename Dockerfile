# syntax=docker/dockerfile:1
# EarlyDrop: CPU-only image. Everything here is classical ML on ~400 rows; no GPU needed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=120 MPLBACKEND=Agg
WORKDIR /app

COPY requirements.txt .
# Slow or flaky networks: the cache mount keeps finished downloads across failed builds,
# and resume-retries continues a broken download instead of restarting it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --retries 10 --resume-retries 10 -r requirements.txt

COPY earlydrop ./earlydrop
COPY data/real ./data/real
COPY run_all.py .
COPY deck ./deck
# Run the whole pipeline at build time so the container starts with results ready.
RUN python run_all.py

COPY app ./app
COPY .streamlit ./.streamlit

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
