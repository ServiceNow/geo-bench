
FROM nvidia/cuda:10.2-cudnn7-devel as base

# >>> Base system configuration
ENV DEBIAN_FRONTEND=noninteractive
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
    python3.7 \
    python3.7-dev \
    python3-pip && \
    apt-get clean -y
# -- Enable GPU access from ssh login into the Docker container
RUN echo "ldconfig" >> /etc/profile

# >>> Python configuration and dependencies
# -- Install requirements
COPY requirements.txt /tmp/requirements.txt
RUN python3.7 -m pip install --upgrade pip setuptools wheel
RUN python3.7 -m pip install --default-timeout=100 -r /tmp/requirements.txt

# -- Make Python 3 the default
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.7 10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 10