# import os
from celery import Celery
# from dotenv import load_dotenv
# load_dotenv()

# RABBIT_USER = os.getenv("RABBITMQ_USER")
# RABBIT_PASS = os.getenv("RABBITMQ_PASS")
# RABBIT_HOST = os.getenv("RABBITMQ_HOST")
# RABBIT_PORT = os.getenv("RABBITMQ_PORT")
RABBIT_USER = "guest"
RABBIT_PASS = "guest"
RABBIT_HOST = "localhost"
RABBIT_PORT = "5672"


BROKER_URL = f"amqp://{RABBIT_USER}:{RABBIT_PASS}@{RABBIT_HOST}:{RABBIT_PORT}//"

app = Celery("tasks", broker=BROKER_URL)

import new_face.workers.tasks

# from celery import Celery
# app = Celery(
#     "tasks",
#     broker="pyamqp://guest@localhost//",   
#     # backend="rpc://"                      
# )
# import new_face.workers.tasks