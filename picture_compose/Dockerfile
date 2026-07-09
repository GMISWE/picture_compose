# syntax=docker/dockerfile:1.6
#
# Image Compose — GMI AgentBox container image
#
# Build:   docker build --platform linux/amd64 -t gracedeng87/image-compose:latest .
# Run:     docker run --rm -p 8080:8080 \
#              -e GMI_MAAS_API_KEY=sk-... \
#              gracedeng87/image-compose:latest
#
# On AgentBox, GMI_MAAS_BASE_URL / GMI_MAAS_API_KEY / GMI_MODELS are injected
# at runtime when MaaS integration is enabled in Step 2 of the register wizard.
# The container listens on 8080 to match AgentBox's default port mapping
# (HTTPS/2 :443 → :8080). No networking edits needed in Step 3.

FROM python:3.11-slim

# System deps for Pillow's optional JPEG/PNG support. Curl is included for
# AgentBox health-check probes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY app.py .

# Listen on AgentBox's default internal port. app.py reads $PORT (default 8080).
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False
EXPOSE 8080

# Optional health endpoint AgentBox can hit during boot. Uses $PORT so it
# tracks whatever the container is actually listening on.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT:-8080}/ > /dev/null || exit 1

CMD ["python", "app.py"]
