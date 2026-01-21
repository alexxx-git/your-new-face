from new_face.workers.tasks import test_task

job = test_task.delay("Hello RabbitMQ + Celery!")
print(job.id)