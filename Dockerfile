# base image
FROM usbmd/base:latest

# Do not prompt for input when installing packages
ARG DEBIAN_FRONTEND=noninteractive

# Set pip and poetry environment variables
ENV PIP_CACHE_DIR=/tmp/pip_cache \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_VIRTUALENVS_PATH=/opt/

WORKDIR /vsmc
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN --mount=type=cache,target=$POETRY_CACHE_DIR --mount=type=ssh \
    poetry install --with torch,tensorflow,jax

ENV PYTHONPATH="${PYTHONPATH}:/vsmc"