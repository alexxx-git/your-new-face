import os


class Settings:
    RMQ_USER = os.getenv("RMQ_USER", "student")
    RMQ_PASSWORD = os.getenv("RMQ_PASSWORD", "qwerty")
    RMQ_HOST = os.getenv("RMQ_HOST", "rmq")
    RMQ_PORT = os.getenv("RMQ_PORT", 5672)
    RMQ_URL = f"amqp://{RMQ_USER}:{RMQ_PASSWORD}@{RMQ_HOST}:{RMQ_PORT}/"

    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
    MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "minio")
    MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "minio123")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "faces")
