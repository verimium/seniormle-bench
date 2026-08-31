FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/task-venv/bin:$PATH"

ARG UV_VERSION=0.10.10

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

COPY . /tests/
RUN chmod 755 /tests/test.sh \
    && chmod -R a-w /tests

WORKDIR /app
