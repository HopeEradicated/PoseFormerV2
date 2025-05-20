FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV CUDA_LAUNCH_BLOCKING=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    ca-certificates \
    python3.9 \
    python3.9-dev \
    python3.9-venv && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py && \
    python3.9 /tmp/get-pip.py && \
    rm /tmp/get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY requirements.txt ./

RUN python -m pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

COPY . .

ENTRYPOINT []

CMD ["bash", "-c", "python demo/vis.py --video sample_video.mp4 && python compare_outputs.py"]