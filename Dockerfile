FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgl1 gcc g++ && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY backend ./backend
COPY models ./models
COPY database ./database
COPY memory ./memory
COPY rag ./rag
COPY inference ./inference
COPY audio ./audio
COPY emotion ./emotion
COPY vision ./vision
COPY video ./video
COPY training ./training
COPY evaluation ./evaluation
COPY export ./export
COPY config ./config
RUN pip install --upgrade pip && pip install '.[train,video,speech]'
RUN useradd --create-home --uid 10001 glm && mkdir -p /app/.glm-state /app/artifacts /app/checkpoints /app/data /app/logs /app/config/generated && chown -R glm:glm /app
USER glm
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["sh", "-c", "exec python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
