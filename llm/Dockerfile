# FROM python:3.11-slim
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
WORKDIR /app
#RUN apt-get update 
#&& apt-get install -y git git-lfs python3 python3-pip python3-dev wget software-properties-common ccache && rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git git-lfs python3 python3-pip python3-dev wget software-properties-common ccache && \
    rm -rf /var/lib/apt/lists/*
    
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.1/cmake-3.26.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.26.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.26.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.26.1/bin/* /usr/local/bin

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN git lfs install


COPY requirements.txt llm_api.py .
RUN pip install -r requirements.txt
RUN mkdir -p ./Qwen2.5-VL-7B-Instruct
WORKDIR /app/Qwen2.5-VL-7B-Instruct
COPY ./Qwen2.5-VL-7B-Instruct .
#RUN git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

WORKDIR /app
COPY llm_api.py .

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775

EXPOSE 8100
CMD ["uvicorn", "llm_api:app", "--host", "0.0.0.0", "--port", "8100"]
