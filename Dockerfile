# =============================================================================
# acne-ai-engine - Motor de inferencia CNN y analisis estadistico (FastAPI)
#
# Python 3.11 is pinned here as the specification requires, which also sidesteps
# the study machine's constraint: it only has Python 3.14, where PyTorch has no
# stable wheels yet.
# =============================================================================

FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 is needed by SciPy/NumPy; the rest keeps Pillow able to decode JPEG/PNG.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libjpeg62-turbo zlib1g \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --upgrade pip && pip install .

# Extend with `--build-arg WITH_TORCH=1` once a trained checkpoint exists.
ARG WITH_TORCH=0
RUN if [ "$WITH_TORCH" = "1" ]; then pip install ".[torch]"; fi

RUN useradd --create-home --uid 10001 acne && chown -R acne:acne /app
USER acne

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
