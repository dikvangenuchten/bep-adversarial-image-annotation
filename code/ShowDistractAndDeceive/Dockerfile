FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt update && apt install gcc -y

COPY requirements.txt requirements.txt

RUN pip3 --disable-pip-version-check --no-cache-dir install -r requirements.txt

COPY . .
