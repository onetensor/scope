FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python-is-python3 \
    git curl ca-certificates \
    build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /modded-nanogpt

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY requirements.txt /modded-nanogpt/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade

RUN grep -vE '^torch($|[<=>])' requirements.txt > /tmp/requirements.txt && \
    pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY . /modded-nanogpt

CMD ["bash"]
ENTRYPOINT []
