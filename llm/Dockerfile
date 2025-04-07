# FROM python:3.11-slim
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    git-lfs \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
#RUN pip install https://github.com/ModelCloud/GPTQModel/releases/download/v2.2.0/gptqmodel-2.2.0+cu121torch2.1-cp310-cp310-linux_x86_64.whl

# Install Python dependencies
RUN pip3 install -U pip setuptools wheel && \
    pip3 install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r requirements.txt 
RUN pip3 install --upgrade safetensors

#RUN pip install -v gptqmodel --no-build-isolation
# Copy the rest of the application
COPY . .

# Make check_model.sh executable
RUN chmod +x check_model.sh

# Run check_model.sh during build
RUN ./check_model.sh

# Expose port
EXPOSE 8100

# Command to run the application
CMD ["uvicorn", "llm_api:app", "--host", "0.0.0.0", "--port", "8100"]
