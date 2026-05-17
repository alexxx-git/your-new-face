import os


class Settings:
    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "5"))
    MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    RMQ_USER = os.getenv("RMQ_USER", "student")
    RMQ_PASSWORD = os.getenv("RMQ_PASSWORD", "qwerty")
    RMQ_HOST = os.getenv("RMQ_HOST", "rmq")
    RMQ_PORT = os.getenv("RMQ_PORT", "5672")
    RMQ_URL = f"amqp://{RMQ_USER}:{RMQ_PASSWORD}@{RMQ_HOST}:{RMQ_PORT}/"

    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
    MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "minio")
    MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "minio123")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "faces")
    MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

    POSTGRES_DB = os.getenv("POSTGRES_DB", "your_new_face")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "your_new_face")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "change_me")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
    )
    RMQ_MANAGEMENT_PORT = os.getenv("RMQ_MANAGEMENT_PORT", "15672")
    RMQ_QUEUE_NAME = os.getenv("RMQ_QUEUE_NAME", "celery")
    RMQ_MANAGEMENT_URL = f"http://{RMQ_HOST}:{RMQ_MANAGEMENT_PORT}"
