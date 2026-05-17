from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from api.celery_app import app as celery_app
from api.config import Settings
from api.db import get_db_session
from api.repositories import (
    count_generation_metrics,
    create_task as create_task_record,
    get_latest_generation_metric,
    get_task,
    get_task_queue_info,
    mark_task_queued,
    task_to_response,
)
from api.storage import get_minio_client, upload_bytes
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
    if not 20 <= target_age <= 60:
        raise HTTPException(status_code=400, detail="Target age must be from 20 to 60.")

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

    celery_task = celery_app.send_task("workers.tasks.process_face", args=[str(db_task.id)])
    queued_task = await mark_task_queued(session, db_task.id, celery_task.id)
    queue_info = await get_task_queue_info(session, queued_task or db_task)

    return {
        "task_id": str(db_task.id),
        "celery_task_id": celery_task.id,
        "status": "PENDING",
        "target_age": target_age,
        **queue_info,
    }


@app.get("/api/tasks/{task_id}")
async def get_task_status(
    task_id: str, session: AsyncSession = Depends(get_db_session)
):
    task = await get_task(session, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    queue_info = await get_task_queue_info(session, task)
    return task_to_response(task, queue_info)


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


@app.get("/api/generation-metrics", response_class=PlainTextResponse)
async def generation_metrics(session: AsyncSession = Depends(get_db_session)):
    total = await count_generation_metrics(session)
    latest = await get_latest_generation_metric(session)

    inference_seconds = 0.0
    age_mae = 0.0
    target_age = 0.0
    result_age = 0.0
    face_similarity = 0.0
    quality_score = 0.0
    lpips = 0.0

    if latest is not None:
        metrics_json = latest.metrics_json or {}
        nested_metrics = metrics_json.get("metrics") or {}
        inference_seconds = _metric_float(metrics_json.get("inference_seconds"))
        age_mae = _metric_float(nested_metrics.get("age_mae"))
        lpips = _metric_float(nested_metrics.get("lpips"))
        target_age = _metric_float(latest.target_age)
        result_age = _metric_float(latest.result_age)
        face_similarity = _metric_float(latest.face_similarity)
        quality_score = _metric_float(latest.quality_score)

    queue_stats = await _collect_queue_stats()

    lines = [
        "# HELP your_new_face_generation_total Total completed generations.",
        "# TYPE your_new_face_generation_total counter",
        f"your_new_face_generation_total {total}",
        "# HELP your_new_face_latest_inference_seconds Last generation inference time.",
        "# TYPE your_new_face_latest_inference_seconds gauge",
        f"your_new_face_latest_inference_seconds {inference_seconds}",
        "# HELP your_new_face_latest_face_similarity Last face similarity score.",
        "# TYPE your_new_face_latest_face_similarity gauge",
        f"your_new_face_latest_face_similarity {face_similarity}",
        "# HELP your_new_face_latest_quality_score Last composite quality score.",
        "# TYPE your_new_face_latest_quality_score gauge",
        f"your_new_face_latest_quality_score {quality_score}",
        "# HELP your_new_face_latest_lpips Last LPIPS perceptual distance, lower is better.",
        "# TYPE your_new_face_latest_lpips gauge",
        f"your_new_face_latest_lpips {lpips}",
        "# HELP your_new_face_latest_age_mae Last age mean absolute error.",
        "# TYPE your_new_face_latest_age_mae gauge",
        f"your_new_face_latest_age_mae {age_mae}",
        "# HELP your_new_face_latest_target_age Last requested target age.",
        "# TYPE your_new_face_latest_target_age gauge",
        f"your_new_face_latest_target_age {target_age}",
        "# HELP your_new_face_latest_result_age Last predicted result age.",
        "# TYPE your_new_face_latest_result_age gauge",
        f"your_new_face_latest_result_age {result_age}",
        "# HELP your_new_face_queue_messages Messages waiting in the processing queue.",
        "# TYPE your_new_face_queue_messages gauge",
        f"your_new_face_queue_messages {queue_stats['messages']}",
    ]
    return "\n".join(lines) + "\n"


@app.get("/api/queue/stats")
async def get_queue_stats():
    stats = await _collect_queue_stats()
    return {"queue": Settings.RMQ_QUEUE_NAME, **stats}


async def _collect_queue_stats() -> dict[str, int]:
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
        "messages": int(queue_messages or 0),
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


def _metric_float(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
