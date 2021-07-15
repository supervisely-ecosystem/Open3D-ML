
FROM python:3.8.11

RUN python3 -m pip install --no-cache-dir tensorflow==2.4.1

RUN pip install open3d
RUN pip install supervisely
#
## TODO: This image has python 3.6.9 onboard, need update to >=3.8 for latest sly package
#RUN pip3 install supervisely==6.1.50
#RUN pip3 install open3d
#
#
#RUN apt update && apt install -y git
#RUN apt install -y libgl1-mesa-glx
#
#WORKDIR '/'
#RUN git clone https://github.com/visignibraem/waymo_tools.git
#WORKDIR '/waymo_tools/supervisely/src'
