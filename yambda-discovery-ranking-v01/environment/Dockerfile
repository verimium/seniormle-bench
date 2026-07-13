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

# train_listens.parquet is not committed: at 304 MB it exceeds GitHub's 100 MB
# per-file limit. Fetched here at build time and verified against the sha256 of
# the split build_task.py produces, so the graded environment stays reproducible.
#
# This file is derived from the Yambda-5B dataset (Copyright Yandex, Apache-2.0,
# https://huggingface.co/datasets/yandex/yambda) and has been MODIFIED:
# subsampled, re-split, re-keyed, and reformatted as a task fixture. It is not
# the original dataset. See public/data/DATA_NOTICE.md for full attribution.
ARG TRAIN_LISTENS_URL=https://d22tzsb3bb54ay.cloudfront.net/yambda-discovery-ranking-v01/train_listens.parquet
ARG TRAIN_LISTENS_SHA256=d51a0cbaf0db492c3ce63fc68072f75529203f445da3cf87ec59d805e5329e17
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL --retry 5 --retry-all-errors --retry-delay 5 \
        -o /task/data/train_listens.parquet "$TRAIN_LISTENS_URL" \
    && echo "$TRAIN_LISTENS_SHA256  /task/data/train_listens.parquet" | sha256sum -c -

RUN useradd --create-home --uid 1000 agent \
    && mkdir -p /app/solution \
    && chown -R agent:agent /app \
    && chmod -R a-w /task /opt/task-venv

WORKDIR /app
