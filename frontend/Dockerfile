FROM python:3.12-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir streamlit

COPY app.py /app/app.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]
