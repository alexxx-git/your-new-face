from api.celery_app import app
import time


@app.task()
def test_task(data):
    print(f"start: {data}")
    time.sleep(2)  # workings
    result = f"finish: {data}"
    print(result)
    return result


@app.task
def process_face(face_data: dict):
    # Имитируем тяжелую задачу
    print(f"Processing face: {face_data}")
    time.sleep(5)  # имитация работы
    return {"status": "success", "face_id": "abc123"}
