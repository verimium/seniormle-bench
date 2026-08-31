FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/task-venv/bin:$PATH"

ARG CODEX_VERSION=0.147.0
ARG UV_VERSION=0.10.10

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && export CODEX_HOME=/opt/codex-home \
    && export CODEX_INSTALL_DIR=/usr/local/bin \
    && export CODEX_RELEASE="$CODEX_VERSION" \
    && export CODEX_NON_INTERACTIVE=1 \
    && curl -fsSL https://chatgpt.com/codex/install.sh | sh \
    && codex --version | grep -F "codex-cli $CODEX_VERSION" \
    && chmod -R a+rX /opt/codex-home

COPY requirements.txt /tmp/task-requirements.txt
RUN /usr/local/bin/python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        "uv==$UV_VERSION" \
    && /usr/local/bin/python -m venv /opt/task-venv \
    && uv pip sync \
        --python /opt/task-venv/bin/python \
        --torch-backend cpu \
        /tmp/task-requirements.txt \
    && /usr/local/bin/python -m pip uninstall --yes uv \
    && rm /tmp/task-requirements.txt

COPY public/data/ /task/data/
COPY public/evaluate_public.py public/ranking_evaluation.py /task/
COPY public/variant/ /task/
COPY requirements.txt /task/requirements.txt

RUN useradd --create-home --uid 1000 agent \
    && mkdir -p /app/solution \
    && chown -R agent:agent /app \
    && chmod -R a+rX,a-w /task /opt/task-venv

WORKDIR /app
