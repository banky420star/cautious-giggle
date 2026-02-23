FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 9090 5678
CMD ["python", "Python/Server_AGI.py"]
