FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/task-venv/bin:$PATH"

COPY requirements.txt /tmp/task-requirements.txt
RUN python -m venv /opt/task-venv \
    && /opt/task-venv/bin/python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --requirement /tmp/task-requirements.txt \
    && rm /tmp/task-requirements.txt

COPY public/ /task/
COPY requirements.txt /task/requirements.txt

RUN useradd --create-home --uid 1000 agent \
    && mkdir -p /app/solution \
    && chown -R agent:agent /app \
    && chmod -R a+rX,a-w /task /opt/task-venv

WORKDIR /app
