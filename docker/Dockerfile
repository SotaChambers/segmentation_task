ARG PYTHON_ENV=python:3.10
ARG PROJECT_ROOT=src/segmentation_task

FROM ${PYTHON_ENV}-slim as python-base

# ROOTを環境変数にしておく(importできる)
ENV PYTHONPATH="/src/segmentation_task/:$PYTHONPATH"

WORKDIR /src/segmentation_task
COPY pyproject.toml poetry.lock ./

RUN \
    apt-get update -y \
    # DockerコンテナからOpenCVを使うと"ImportError: libGL.so.1: cannot open shared object file: No such file or directory"と言われるので，libgl1-mesa-devを入れる
    && apt-get install -y make libopencv-dev \
    && pip install -U pip poetry \
    # 仮想環境を作成しない設定(コンテナ前提のため，仮想環境を作らない)
    && poetry config virtualenvs.create false \
    && poetry install