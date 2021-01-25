FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel
RUN apt update && apt upgrade -y
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install tqdm opencv-python torchfile
RUN mkdir /style-transfer
WORKDIR /style-transfer
