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
    && rm -rf /var/lib/apt/lists/*

RUN curl -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz | tar xz && \
    cd ta-lib && \
    ./configure --build=x86_64-pc-linux-gnu --prefix=/usr/local --enable-shared && \
    make && make install && \
    ln -s /usr/local/lib/libta_lib.so /usr/local/lib/libta-lib.so && \
    ln -s /usr/local/lib/libta_lib.a /usr/local/lib/libta-lib.a && \
    cd .. && rm -rf ta-lib

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
ENV CPATH=/usr/local/include:$CPATH

RUN pip install --upgrade pip
RUN pip install vectorbt TA-Lib

WORKDIR /workspace
