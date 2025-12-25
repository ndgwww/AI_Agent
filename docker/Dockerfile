FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PIP_DEFAULT_TIMEOUT=120

# 换源加速（Ubuntu）
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    apt-get update

# 安装 Python 3.10 和工具（Ubuntu 22.04 默认版本）
RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    git \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 创建 python 符号链接
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 配置 pip 使用国内源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip config set global.timeout 120

# 验证 Python 和 pip
RUN python --version && pip --version

# 升级 pip
RUN pip install --upgrade pip setuptools wheel

# 安装 PyTorch（CUDA 12.1）
RUN pip install --no-cache-dir torch torchvision torchaudio

# 安装 RWKV 和依赖
RUN pip install --no-cache-dir \
    rwkv>=0.8.28 \
    numpy \
    tokenizers \
    ninja \
    packaging
RUN pip install bitsandbytes einops triton rwkv-fla rwkv transformers GPUtil plotly datasets --upgrade
WORKDIR /workspace

ENV RWKV_V7_ON=1 \
    RWKV_JIT_ON=1 \
    RWKV_CUDA_ON=1

CMD ["/bin/bash"]
