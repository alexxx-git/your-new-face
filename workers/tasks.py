import asyncio
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from api.celery_app import app
from api.db import async_session
from api.repositories import (
    get_task,
    mark_task_failed,
    mark_task_processing,
    mark_task_success,
)
from api.storage import download_bytes, upload_bytes
import time


@app.task()
def test_task(data):
    print(f"start: {data}")
    time.sleep(2)  # workings
    result = f"finish: {data}"
    print(result)
    return result


@app.task
def process_face(task_id: str):
    return asyncio.run(process_face_async(task_id))


async def process_face_async(task_id: str):
    print(f"Processing task: {task_id}")
    await asyncio.sleep(5)  # временная имитация тяжелой обработки

    try:
        async with async_session() as session:
            task = await mark_task_processing(session, task_id)
            if task is None:
                raise ValueError(f"Task not found: {task_id}")

            source_object = task.source_object
            content_type = task.content_type or "image/jpeg"

        source_bytes = download_bytes(source_object)

        with Image.open(BytesIO(source_bytes)) as image:
            mirrored = ImageOps.mirror(image)
            image_format = _format_from_content_type(content_type)

            if image_format == "JPEG" and mirrored.mode in {"RGBA", "P"}:
                mirrored = mirrored.convert("RGB")

            output = BytesIO()
            mirrored.save(output, format=image_format)

        source_path = Path(source_object)
        result_object = f"outputs/{source_path.stem}-mirrored{source_path.suffix}"
        upload_bytes(result_object, output.getvalue(), content_type)

        async with async_session() as session:
            await mark_task_success(session, task_id, result_object)

        return {
            "status": "success",
            "task_id": task_id,
            "source_object": source_object,
            "result_object": result_object,
            "result_url": f"/api/images/{result_object}",
        }
    except Exception as exc:
        async with async_session() as session:
            await mark_task_failed(session, task_id, str(exc))
        raise


def _format_from_content_type(content_type: str) -> str:
    formats = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
    }
    return formats.get(content_type, "JPEG")
