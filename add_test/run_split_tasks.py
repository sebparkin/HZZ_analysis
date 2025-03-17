from split_tasks import iteration
from celery.result import AsyncResult

# Submit 1000 tasks
results = [iteration.delay(i) for i in range(10)]

# Collect and sum results
total = [result.get() for result in results]

print(f"Final Total: {total}")