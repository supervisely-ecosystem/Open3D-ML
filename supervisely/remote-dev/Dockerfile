FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y curl

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# ubuntu 20 has python 3.8
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools


RUN ln -s $(which python3) /usr/local/bin/python

RUN apt-get -y install libcusolver-11-0
#RUN export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.0/lib64
RUN echo '/usr/local/cuda-11.2/lib64' >> /etc/ld.so.conf.d/nvidia.conf
RUN echo '/usr/local/cuda-11.0/lib64' >> /etc/ld.so.conf.d/nvidia.conf
RUN ldconfig

RUN pip install tensorflow==2.4.1
RUN pip install open3d
RUN pip install supervisely

RUN apt update && apt install -y git
RUN apt install -y libgl1-mesa-glx

WORKDIR '/'
RUN git clone https://github.com/supervisely-ecosystem/Open3D-ML.git
WORKDIR '/Open3D-ML'
RUN git checkout sly
WORKDIR '/Open3D-ML/supervisely/src'
