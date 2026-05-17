import asyncio
import tempfile
from time import perf_counter
from pathlib import Path

from api.celery_app import app
from api.db import async_session
from api.repositories import (
    create_generation_metric,
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

    try:
        async with async_session() as session:
            task = await mark_task_processing(session, task_id)
            if task is None:
                raise ValueError(f"Task not found: {task_id}")

            source_object = task.source_object
            content_type = task.content_type or "image/jpeg"
            target_age = task.target_age

        source_bytes = download_bytes(source_object)

        source_path = Path(source_object)
        with tempfile.TemporaryDirectory(prefix=f"ynf-{task_id}-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / f"input{source_path.suffix}"
            output_path = tmp_path / f"output{source_path.suffix}"
            input_path.write_bytes(source_bytes)

            from new_face.test_single_pass import run_single_pass

            inference_started_at = perf_counter()
            generation = run_single_pass(
                input_path=str(input_path),
                output_path=str(output_path),
                target_age=int(target_age or 40),
                task_id=task_id,
            )
            inference_seconds = perf_counter() - inference_started_at

            result_bytes = output_path.read_bytes()
            result_object = f"outputs/{source_path.stem}-generated{source_path.suffix}"
            upload_bytes(result_object, result_bytes, content_type)

            metrics = generation.get("metrics") or {}
            source_age = _safe_int(generation.get("source_age_used"))
            result_age = _safe_int(metrics.get("predicted_age"))
            async with async_session() as session:
                await create_generation_metric(
                    session,
                    task_id=task_id,
                    target_age=target_age,
                    source_age=source_age,
                    result_age=result_age,
                    face_similarity=metrics.get("cosine_similarity"),
                    quality_score=metrics.get("composite_score"),
                    metrics_json=_json_safe({
                        **(metrics.get("metrics_json") or {}),
                        "metrics": metrics,
                        "prompt": generation.get("prompt"),
                        "negative_prompt": generation.get("negative_prompt"),
                        "run": generation.get("run"),
                        "source_age_meta": generation.get("source_age_meta"),
                        "inference_seconds": inference_seconds,
                    }),
                )

        async with async_session() as session:
            await mark_task_success(session, task_id, result_object)

        return {
            "status": "success",
            "task_id": task_id,
            "source_object": source_object,
            "result_object": result_object,
            "result_url": f"/api/images/{result_object}",
            "target_age": target_age,
            "inference_seconds": inference_seconds,
        }
    except Exception as exc:
        async with async_session() as session:
            await mark_task_failed(session, task_id, str(exc))
        raise


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
