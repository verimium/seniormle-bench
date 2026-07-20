FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/task-venv/bin:$PATH" \
    REPAIR_TEST_UID=65534 \
    REPAIR_TEST_GID=65534

COPY requirements.txt /tmp/task-requirements.txt
RUN python -m venv /opt/task-venv \
    && /opt/task-venv/bin/python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --requirement /tmp/task-requirements.txt \
    && rm /tmp/task-requirements.txt

COPY . /tests/
RUN chmod 755 /tests/test.sh \
    && chmod -R a-w /tests \
    && chmod 500 /tests/private

WORKDIR /app
