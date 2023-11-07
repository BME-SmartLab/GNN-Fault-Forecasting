FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    wget \
    tmux \
    vim \
    htop \
    openssh-server \
    zip \
    unzip \
    build-essential

RUN rm -rf /var/lib/apt/lists/*

RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html torch_geometric
RUN pip install debugpy pytest tensorboardX matplotlib seaborn pandas

RUN useradd -m -s /bin/bash myusername

WORKDIR /home/myusername

USER myusername

RUN git clone https://github.com/BME-SmartLab/GNN-SLA-Forecast.git && cd GNN-SLA-Forecast

WORKDIR /home/myusername/GNN-SLA-Forecast
