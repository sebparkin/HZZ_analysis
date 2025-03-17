FROM python:latest

RUN pip install celery

COPY . .

CMD ["celery", "-A", "split_tasks", "worker", "--loglevel=info"]
