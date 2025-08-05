FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    liblzma-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libnss3-dev \
    libx11-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libz-dev \
    pkg-config \
    cmake \
    automake \
    autoconf \
    libtool \
    dpkg-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/TA-Lib/ta-lib/releases/download/v0.6.1/ta-lib-0.6.1-src.tar.gz && \
    tar -xzf ta-lib-0.6.1-src.tar.gz && \
    cd ta-lib-0.6.1 && \
    ./configure --prefix=/usr/local --enable-shared && \
    make && make install

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
ENV CPATH=/usr/local/include:$CPATH

# Install Python wrapper
RUN python3 -m pip install --no-cache-dir ta-lib

# Optional: Verify symbol
RUN nm -D /usr/local/lib/libta_lib.so | grep TA_AVGDEV_Lookback || echo "Symbol not found"

# Install other Python packages
RUN pip install --upgrade pip
RUN pip install vectorbt

WORKDIR /workspace
