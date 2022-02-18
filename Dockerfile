
FROM nvidia/cuda:10.2-cudnn7-devel as base

# >>> Base system configuration
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf-8

# -- Install system packages
RUN apt-get update  -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    unrar \
    unzip \
    libnvinfer6=6.0.1-1+cuda10.1 \
    libnvinfer-dev=6.0.1-1+cuda10.1 \
    libnvinfer-plugin6=6.0.1-1+cuda10.1 \
    git && \
    apt-get clean -y
# -- Install Python
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update  -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip && \
    apt-get clean -y
# -- Enable GPU access from ssh login into the Docker container
RUN echo "ldconfig" >> /etc/profile

RUN python3.9 -m pip install --upgrade pip setuptools wheel

# -- Make Python 3 the default
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.9 10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 10

# Install the specified `poetry` version.
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | \
        POETRY_HOME=/usr/local/poetry python - --version 1.1.12 \
    && chmod 755 /usr/local/poetry/bin/poetry \
    && ln -sf /usr/local/poetry/bin/poetry /usr/local/bin/poetry

# >>> Python configuration and dependencies
# -- Install requirements
COPY pyproject.toml ./
COPY poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --no-root
