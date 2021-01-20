FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt

COPY download.sh download.sh

RUN apt-get udpate

RUN apt-get install -y wget

RUN apt-get install -y unzip

RUN pip install -r requirements.txt

RUN sh download.sh

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]