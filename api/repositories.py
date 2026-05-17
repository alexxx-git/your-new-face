from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.models import GenerationMetric, Task


async def create_task(
    session: AsyncSession,
    *,
    source_object: str,
    content_type: str,
    original_filename: str | None,
    file_size: int,
    target_age: int | None,
) -> Task:
    task = Task(
        status="UPLOADED",
        source_object=source_object,
        content_type=content_type,
        original_filename=original_filename,
        file_size=file_size,
        target_age=target_age,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


async def get_task(session: AsyncSession, task_id: str | UUID) -> Task | None:
    return await session.get(Task, _task_uuid(task_id))


async def mark_task_queued(
    session: AsyncSession, task_id: str | UUID, celery_task_id: str
) -> Task | None:
    task = await get_task(session, task_id)
    if task is None:
        return None

    task.status = "PENDING"
    task.celery_task_id = celery_task_id
    await session.commit()
    await session.refresh(task)
    return task


async def mark_task_processing(session: AsyncSession, task_id: str | UUID) -> Task | None:
    task = await get_task(session, task_id)
    if task is None:
        return None

    task.status = "PROCESSING"
    task.started_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(task)
    return task


async def mark_task_success(
    session: AsyncSession, task_id: str | UUID, result_object: str
) -> Task | None:
    task = await get_task(session, task_id)
    if task is None:
        return None

    task.status = "SUCCESS"
    task.result_object = result_object
    task.finished_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(task)
    return task


async def mark_task_failed(
    session: AsyncSession, task_id: str | UUID, error_message: str
) -> Task | None:
    task = await get_task(session, task_id)
    if task is None:
        return None

    task.status = "FAILURE"
    task.error_message = error_message
    task.finished_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(task)
    return task


async def create_generation_metric(
    session: AsyncSession,
    *,
    task_id: str | UUID,
    target_age: int | None,
    source_age: int | None,
    result_age: int | None,
    face_similarity: float | None,
    quality_score: float | None,
    metrics_json: dict | None,
) -> GenerationMetric:
    metric = GenerationMetric(
        task_id=_task_uuid(task_id),
        target_age=target_age,
        source_age=source_age,
        result_age=result_age,
        face_similarity=face_similarity,
        quality_score=quality_score,
        metrics_json=metrics_json,
    )
    session.add(metric)
    await session.commit()
    await session.refresh(metric)
    return metric


async def get_latest_generation_metric(session: AsyncSession) -> GenerationMetric | None:
    result = await session.execute(
        select(GenerationMetric).order_by(GenerationMetric.created_at.desc()).limit(1)
    )
    return result.scalar_one_or_none()


async def count_generation_metrics(session: AsyncSession) -> int:
    result = await session.execute(select(func.count(GenerationMetric.id)))
    return int(result.scalar_one())


async def get_task_queue_info(session: AsyncSession, task: Task) -> dict:
    if task.status == "PROCESSING":
        return {"queue_position": 0, "queue_ahead": 0}

    if task.status != "PENDING":
        return {"queue_position": None, "queue_ahead": None}

    pending_before = await session.execute(
        select(func.count(Task.id)).where(
            Task.status == "PENDING",
            Task.created_at < task.created_at,
        )
    )
    processing_now = await session.execute(
        select(func.count(Task.id)).where(Task.status == "PROCESSING")
    )

    pending_before_count = int(pending_before.scalar_one())
    processing_count = int(processing_now.scalar_one())

    return {
        "queue_position": pending_before_count + 1,
        "queue_ahead": pending_before_count + processing_count,
    }


def task_to_response(task: Task, queue_info: dict | None = None) -> dict:
    result = None
    if task.status == "SUCCESS" and task.result_object:
        result = {
            "source_object": task.source_object,
            "result_object": task.result_object,
            "result_url": f"/api/images/{task.result_object}",
        }
    elif task.status == "FAILURE":
        result = {"error": task.error_message or "Task failed"}

    response = {
        "task_id": str(task.id),
        "celery_task_id": task.celery_task_id,
        "status": task.status,
        "target_age": task.target_age,
        "result": result,
    }
    if queue_info is not None:
        response.update(queue_info)
    return response


def _task_uuid(task_id: str | UUID) -> UUID:
    if isinstance(task_id, UUID):
        return task_id
    return UUID(task_id)
