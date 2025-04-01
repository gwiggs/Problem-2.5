# streamlit_app/Dockerfile
FROM python:3.11-slim


COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#COPY . /app #uncomment for production
WORKDIR /app

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
