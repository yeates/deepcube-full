FROM registry.codeocean.com/codeocean/miniconda3:4.5.11-python3.7-cuda9.2-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      "build-essential=12.4ubuntu1" \
      "libboost-all-dev=1.65.1.0ubuntu1" \
      "libboost-dev=1.65.1.0ubuntu1" \
    && rm -rf /var/lib/apt/lists/*

RUN conda install --yes \
      python==2.7.5 \
      tensorflow-gpu==1.8.0 \
    && conda clean --yes --all

RUN pip install --upgrade \
      dm-sonnet==1.10 \
      matplotlib==2.2.3

COPY postInstall /
RUN /postInstall
