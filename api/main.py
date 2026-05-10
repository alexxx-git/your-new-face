from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from api.celery_app import app as celery_app
from api.config import Settings
from api.db import get_db_session
from api.repositories import (
    create_task as create_task_record,
    get_task,
    mark_task_queued,
    task_to_response,
)
from api.storage import get_minio_client, upload_bytes
from workers.tasks import process_face
import psutil
import requests
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Your New Face API", docs_url="/api/docs", openapi_url="/api/openapi.json"
)
Instrumentator().instrument(app).expose(app, endpoint="/api/metrics")
# CORS для фронта
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    face_path: str
    model: str = "deepface"


ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@app.get("/api/health")
async def health():
    return {"status": "ok", "celery": "ready"}


@app.post("/api/tasks")
async def create_task(request: TaskRequest):
    raise HTTPException(
        status_code=400,
        detail="Use /api/tasks/upload to create an image processing task.",
    )


@app.post("/api/tasks/upload")
async def upload_task(
    file: UploadFile = File(...),
    target_age: int = Form(20),
    session: AsyncSession = Depends(get_db_session),
):
    if not 20 <= target_age <= 100:
        raise HTTPException(status_code=400, detail="Target age must be from 20 to 100.")

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG, and WEBP images are supported.",
        )

    content = await file.read()
    if len(content) > Settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Max size is {Settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    suffix = ALLOWED_IMAGE_TYPES[file.content_type]
    original_name = f"inputs/{uuid4().hex}{suffix}"
    upload_bytes(original_name, content, file.content_type)

    db_task = await create_task_record(
        session,
        source_object=original_name,
        content_type=file.content_type,
        original_filename=file.filename,
        file_size=len(content),
        target_age=target_age,
    )

    celery_task = process_face.delay(str(db_task.id))
    await mark_task_queued(session, db_task.id, celery_task.id)

    return {
        "task_id": str(db_task.id),
        "celery_task_id": celery_task.id,
        "status": "PENDING",
        "target_age": target_age,
    }


@app.get("/api/tasks/{task_id}")
async def get_task_status(
    task_id: str, session: AsyncSession = Depends(get_db_session)
):
    task = await get_task(session, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return task_to_response(task)


@app.get("/api/images/{object_name:path}")
async def get_image(object_name: str):
    client = get_minio_client()
    try:
        stat = client.stat_object(Settings.MINIO_BUCKET, object_name)
        response = client.get_object(Settings.MINIO_BUCKET, object_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Image not found") from exc

    def iter_image():
        try:
            for chunk in response.stream(32 * 1024):
                yield chunk
        finally:
            response.close()
            response.release_conn()

    filename = Path(object_name).name
    headers = {"Content-Disposition": f'inline; filename="{filename}"'}
    return StreamingResponse(
        iter_image(),
        media_type=stat.content_type or "application/octet-stream",
        headers=headers,
    )


@app.get("/api/system")
async def get_system_stats():
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory": psutil.virtual_memory()._asdict(),
        "disk": psutil.disk_usage("/")._asdict(),
    }


@app.get("/api/queue/stats")
async def get_queue_stats():
    inspect = celery_app.control.inspect()
    active = inspect.active() or {}
    reserved = inspect.reserved() or {}
    scheduled = inspect.scheduled() or {}
    queue_messages = None
    try:
        response = requests.get(
            f"{Settings.RMQ_MANAGEMENT_URL}/api/queues/%2F/{Settings.RMQ_QUEUE_NAME}",
            auth=(Settings.RMQ_USER, Settings.RMQ_PASSWORD),
            timeout=3,
        )
        response.raise_for_status()
        queue_messages = response.json().get("messages")
    except requests.RequestException:
        queue_messages = None
    return {
        "queue": Settings.RMQ_QUEUE_NAME,
        "messages": queue_messages,
        "active_tasks": sum(len(tasks) for tasks in active.values()),
        "reserved_tasks": sum(len(tasks) for tasks in reserved.values()),
        "scheduled_tasks": sum(len(tasks) for tasks in scheduled.values()),
    }


@app.get("/api/ready")
async def ready():
    checks = {
        "api": "ok",
        "celery": "unknown",
        "rabbitmq": "unknown",
    }
    try:
        ping = celery_app.control.ping(timeout=1)
        checks["celery"] = "ok" if ping else "unavailable"
    except Exception:
        checks["celery"] = "unavailable"
    return checks
