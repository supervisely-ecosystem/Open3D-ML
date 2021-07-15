FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN pip3 install --upgrade pip
RUN pip3 install waymo-open-dataset-tf-2-1-0

# TODO: This image has python 3.6.9 onboard, need update to >=3.8 for latest sly package
RUN pip3 install supervisely==6.1.50
RUN pip3 install open3d


RUN apt update && apt install -y git
RUN apt install -y libgl1-mesa-glx

WORKDIR '/'
RUN git clone https://github.com/visignibraem/waymo_tools.git
WORKDIR '/waymo_tools/supervisely/src'
