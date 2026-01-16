# add_task.py
from rq import Queue
from redis import Redis
from test_task import mytask

# Подключение
redis_conn = Redis(host='127.0.0.1', port=6379)
q = Queue('first_1111', connection=redis_conn)

# Добавить задачу
job = q.enqueue(mytask, "ПРИВЕТ МИР")

print(f"✅ Задача добавлена: {job.id}")
print(f"В очереди: {q.count} задач")