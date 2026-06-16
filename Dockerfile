# Python runtime for shadowbox.
# Pinned to a specific Debian / Python combo for reproducibility.
FROM python:3.13-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    U2NET_HOME=/models

# Runtime deps for scikit-image / Pillow / onnxruntime. apt-get upgrade pulls
# fresh security patches that landed in Debian after the base image was
# published — keeps the image current without waiting for the base tag to roll.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       libgl1 \
       libglib2.0-0 \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache the U²-Net model in its own layer — independent of the project files
# so config edits don't force a 170MB re-download. We install just rembg here
# (deps follow in the next layer); the full project install replaces it.
RUN pip install --upgrade pip \
    && pip install "rembg[cpu]>=2.0.50" \
    && python -c "from rembg import new_session; new_session()"

# Install dependencies next so source edits don't bust the deps layer cache.
COPY pyproject.toml README.md ./
RUN mkdir -p src/shadowbox && touch src/shadowbox/__init__.py \
    && pip install .[web,dev]

COPY src ./src
COPY tests ./tests
RUN pip install --no-deps -e .

WORKDIR /work

ENTRYPOINT ["shadowbox"]
CMD ["--help"]

EXPOSE 8000
