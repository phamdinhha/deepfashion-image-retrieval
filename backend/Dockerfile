FROM python:3.8-slim

WORKDIR /app

RUN pip install --upgrade pip

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY download.sh download.sh

RUN sh download.sh

COPY . .

EXPOSE 8080

CMD ["python", "api.py"]