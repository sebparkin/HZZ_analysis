from celery import Celery, chord

app = Celery('split_tasks', broker = 'pyamqp://guest@rabbitmq//', backend = 'rpc://')
total = 0

@app.task
def iteration(i):
    return 1

print(total)
