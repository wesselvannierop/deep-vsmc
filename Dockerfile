# base image
FROM ubuntu:22.04

# Do not prompt for input when installing packages
ARG DEBIAN_FRONTEND=noninteractive

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Set pip cache directory
ENV PIP_CACHE_DIR=/tmp/pip_cache

# Set poetry version and venv path
ENV POETRY_VERSION=2.1.3 \
    POETRY_VENV=/opt/poetry-venv \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install python, pip, git, opencv dependencies, ffmpeg, imagemagick, and ssh keyscan github
RUN apt-get update && apt-get install -y python3 python3-pip git python3-tk python3-venv \
                       libsm6 libxext6 libxrender-dev libqt5gui5 \
                       ffmpeg imagemagick && \
    python3 -m pip install pip setuptools -U && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    ln -s /usr/bin/python3 /usr/bin/python

# Install poetry
RUN python3 -m venv $POETRY_VENV && \
    $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}
ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /vsmc

# Copy pyproject.toml, poetry.lock, README.md, and source code
COPY pyproject.toml poetry.lock README.md ./
COPY vsmc ./vsmc

# Install dependencies and vsmc
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install