from io import BytesIO

from minio import Minio

from api.config import Settings


def get_minio_client() -> Minio:
    return Minio(
        Settings.MINIO_ENDPOINT,
        access_key=Settings.MINIO_ROOT_USER,
        secret_key=Settings.MINIO_ROOT_PASSWORD,
        secure=Settings.MINIO_SECURE,
    )


def ensure_bucket() -> None:
    client = get_minio_client()
    if not client.bucket_exists(Settings.MINIO_BUCKET):
        client.make_bucket(Settings.MINIO_BUCKET)


def upload_bytes(object_name: str, data: bytes, content_type: str) -> None:
    ensure_bucket()
    client = get_minio_client()
    client.put_object(
        Settings.MINIO_BUCKET,
        object_name,
        BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


def download_bytes(object_name: str) -> bytes:
    client = get_minio_client()
    response = client.get_object(Settings.MINIO_BUCKET, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()
