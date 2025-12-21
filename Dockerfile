FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python-is-python3 \
    git curl ca-certificates \
    build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /modded-nanogpt

COPY requirements.txt /modded-nanogpt/requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade

COPY . /modded-nanogpt

CMD ["bash"]
ENTRYPOINT []
