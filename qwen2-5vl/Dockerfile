# python 3.8 and above
# pytorch 1.12 and above, 2.0 and above are recommended
# CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)

# based on modelscope docker image
# registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0
# registry.cn-beijing.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0
FROM qwenllm/qwenvl:latest

ARG workdir=/var/app
RUN mkdir -p ${workdir}

RUN git lfs install

WORKDIR ${workdir}
COPY requirements.txt ./

# Install Qwen dependencies
#RUN pip install -r requirements.txt

# Install webUI dependencies
WORKDIR ${workdir}
#RUN pip install -r requirements_web_demo.txt

# Offline mode, check https://huggingface.co/docs/transformers/v4.15.0/installation#offline-mode
#ENV HF_DATASETS_OFFLINE=1
#ENV TRANSFORMERS_OFFLINE=1

# set TZ, make logs dir, and expose port 8080
RUN mkdir -p ${workdir}/logs && chmod 777 ${workdir}/logs
VOLUME /var/app/logs

# create user 20001
RUN useradd -r -m appuser -u 20001 -g 0

#WORKDIR ${workdir}
# copy model
#RUN git clone https://huggingface.co/Qwen/Qwen-VL-Chat
# COPY --chown=20001:20001 Qwen-VL-Chat ./Qwen-VL-Chat
# copy fonts
#ADD --chown=20001:20001 https://github.com/StellarCN/scp_zh/raw/master/fonts/SimSun.ttf ./
# COPY --chown=20001:20001 SimSun.ttf ./
# copy main app
#COPY --chown=20001:20001 web_demo_mm.py ./

EXPOSE 8100
CMD ["python3", "--server-name", "0.0.0.0", "--server-port", "8100"]