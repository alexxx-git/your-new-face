from celery import Celery
from config import Settings
import os

app = Celery(
    "yournewface",
    broker=Settings.RMQ_URL,
    backend="rpc://",
)

app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

app.autodiscover_tasks(["workers"])
