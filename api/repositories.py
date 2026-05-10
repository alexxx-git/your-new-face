from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from api.models import Task


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


def task_to_response(task: Task) -> dict:
    result = None
    if task.status == "SUCCESS" and task.result_object:
        result = {
            "source_object": task.source_object,
            "result_object": task.result_object,
            "result_url": f"/api/images/{task.result_object}",
        }
    elif task.status == "FAILURE":
        result = {"error": task.error_message or "Task failed"}

    return {
        "task_id": str(task.id),
        "celery_task_id": task.celery_task_id,
        "status": task.status,
        "target_age": task.target_age,
        "result": result,
    }


def _task_uuid(task_id: str | UUID) -> UUID:
    if isinstance(task_id, UUID):
        return task_id
    return UUID(task_id)
