from celery import Celery
app = Celery(
    "tasks",
    broker="pyamqp://guest@localhost//",   
    # backend="rpc://"                      
)
import new_face.workers.tasks