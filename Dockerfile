FROM python:3.9-slim
WORKDIR /app
COPY my_requirement.txt ./
RUN pip install -r my_requirement.txt
COPY . .
