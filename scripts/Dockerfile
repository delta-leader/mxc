FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

RUN apt update &&\
  DEBIAN_FRONTEND=noninteractive apt install -y wget cmake libopenblas-dev libblas-dev liblapacke-dev libeigen3-dev libopenmpi-dev python3 python3-matplotlib &&\
  rm -rf /var/lib/apt/lists/*

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV OPENBLAS_NUM_THREADS=1
