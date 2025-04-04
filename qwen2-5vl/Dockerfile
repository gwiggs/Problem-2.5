FROM python:3.11-slim

WORKDIR /app

# RUN apt-get update && apt-get install -y git git-lfs


COPY requirements.custom.txt llm_api.py .
RUN pip install -r requirements.custom.txt
#RUN mkdir -p ./Qwen2-VL-7B-Instruct
#COPY Qwen2-VL-7B-Instruct ./Qwen2-VL-7B-Instruct
#RUN git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

EXPOSE 8100
CMD ["uvicorn", "llm_api:app", "--host", "0.0.0.0", "--port", "8100"]