FROM python:latest

WORKDIR /app

RUN useradd -ms /bin/bash celery

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

RUN chown -R celery:celery /app

USER celery

CMD ["python", "saveplot.py"]
