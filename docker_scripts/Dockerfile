FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl wget git sudo build-essential \
    ca-certificates bzip2 unzip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG DOCKER_NAME
ARG DOCKER_USER_ID
ARG DOCKER_GROUP_ID

RUN groupadd -g $DOCKER_GROUP_ID $DOCKER_NAME && \
    useradd -m -u $DOCKER_USER_ID -g $DOCKER_GROUP_ID -s /bin/bash $DOCKER_NAME

RUN echo "$DOCKER_NAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-$DOCKER_NAME

USER $DOCKER_NAME
ENV HOME=/home/$DOCKER_NAME
ENV PATH=$HOME/miniconda/bin:$PATH

RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh

COPY requirements.txt /app/
RUN $HOME/miniconda/bin/pip install -r requirements.txt

CMD jupyter notebook --ip=0.0.0.0 --port=8890 --NotebookApp.token=''
