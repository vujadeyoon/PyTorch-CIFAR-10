# Dveloper: vujadeyoon
# Email: vujadeyoon@gmail.com
# Github: https://github.com/vujadeyoon
# Personal website: https://vujadeyoon.github.io
#
# Title: Dockerfile
# Description: A Dockerfile for the NVIDIA Container Toolkit for Deep Learning


FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04


LABEL maintainer="vujadeyoon"
LABEL email="vujadeyoon@gmial.com"
LABEL version="1.0"
LABEL description="A Dockerfile for PyTorch-Classification"


ARG GIT_MAINTAINER="vujadeyoon"
ARG GIT_EMAIL="vujadeyoon@gmial.com"


ENV DEBIAN_FRONTEND=noninteractive
ENV GIT_MAINTAINER=${GIT_MAINTAINER}
ENV GIT_EMAIL=${GIT_EMAIL}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ Asia/Seoul


RUN mkdir -p /home/dev/ && \
    mkdir -p /home/deb_packages/


# Install the essential ubuntu packages.
RUN apt-get update &&  \
    apt-get upgrade -y &&  \
    apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        cmake \
        curl \
        ssh \
        sudo \
        tar \
        libcurl3-dev \
        libfreetype6-dev \
        pkg-config \
        ca-certificates \
        libjpeg-dev \
        libpng-dev


# Install the useful ubuntu packages.
RUN apt-get update &&  \
    apt-get install -y \
        eog \
        nautilus \
        imagemagick \
        libreoffice \
        python3-tk \
        pv \
        dialog \
        ffmpeg \
        libgtk2.0-dev \
        python3-matplotlib \
        wget \
        tmux \
        zsh \
        locales \
        ncdu \
        htop \
        zip \
        unzip \
        rsync


# Install git.
RUN apt-get update && \
    apt-get install -y git && \
    git config --global user.name "${GIT_MAINTAINER}" && \
    git config --global user.email "${GIT_EMAIL}"


# Install editors.
RUN apt-get update && \
    apt-get install -y vim && \
    echo "set number" | tee -a ~/.vimrc && \
    echo "set ts=8" | tee -a ~/.vimrc && \
    echo "set sw=4" | tee -a ~/.vimrc && \
    echo "set sts=4" | tee -a ~/.vimrc && \
    echo "set smartindent" | tee -a ~/.vimrc && \
    echo "set cindent" | tee -a ~/.vimrc


# Install Korean language.
RUN apt-get update &&  \
    apt-get install -y \
        fcitx \
        fcitx-hangul \
        fonts-nanum*


# Set the locale.
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


# Install Python.
RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3-pip


# Set Python3.8 as default.
RUN ln -s /usr/bin/python3.8 /usr/bin/python


# Upgarde pip and pip3.
RUN pip3 install --upgrade pip && \
    python3 -m pip install --upgrade pip


# Python3 packages for mathematical functions and plotting
RUN pip3 install \
    wheel \
    opencv-python \
    opencv-contrib-python \
    imgaug \
    ffmpeg-python \
    Pillow \
    imageio \
    matplotlib \
    scikit-image \
    scikit-learn \
    pandas \
    openpyxl \
    plotly \
    seaborn \
    shapely \
    PyQt5 \
    pyvista \
    pyvistaqt


# Python3 packages for monitoring and debugging
RUN pip3 install \
    jupyter \
    wandb \
    py-cpuinfo \
    gpustat \
    getgpu \
    tqdm \
    ipdb \
    icecream


# Other python3 packages
RUN pip3 install \
    scipy \
    Cython \
    prettyprinter \
    colorlog \
    randomcolor \
    future \
    imutils \
    psutil \
    PyYAML \
    pycrypto \
    slack_sdk \
    omegaconf


# Install python3 packages related to the cloud and server.
RUN pip3 install --ignore-installed \
    Flask \
    Flask-RESTful \
    gevent \
    boto3 \
    kubernetes


# Install python3 packages for the deep learning research.
RUN pip3 install \
    dlib \
    PyWavelets \
    pycuda>=2022.2.2 \
    tensorflow==2.13.0 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    kornia==0.7.0 \
    torchinfo \
    ptflops \
    onnx \
    onnxruntime \
    onnxruntime-gpu \
    gradio \
    vujade


# Enroll bash functions to ~/.bashrc.
COPY ./script/container/bash_enroll_func_bashrc.sh /home/dev/bash_enroll_func_bashrc.sh
RUN bash /home/dev/bash_enroll_func_bashrc.sh && \
    rm -rf /home/dev/bash_enroll_func_bashrc.sh


# Update and clean the Ubuntu packages.
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Go to the base directory.
WORKDIR /home/dev/


# Run the command.
CMD [ "/bin/bash" ]

