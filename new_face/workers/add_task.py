from rq import Queue
from redis import Redis
# from test_task import mytask
from new_face.workers.add_job import count_words_at_url
import time
# Подключение
redis_conn = Redis(host='127.0.0.1', port=6379)
q = Queue('first', connection=redis_conn)

# # Добавить задачу
# # job = q.enqueue("test", "ПРИВЕТ МИР")
# job = q.enqueue(mytask, "ПРИВЕТ МИР")

# print(f" Задача добавлена: {job.result}")


job = q.enqueue(count_words_at_url, 'http://nvie.com')
print(job.id) 
print(job.return_value())   # => None  # Changed to job.return_value() in RQ >= 1.12.0

# Now, wait a while, until the worker is finished

# print(job.result)   # => 889  # Changed to job.return_value() in RQ >= 1.12.