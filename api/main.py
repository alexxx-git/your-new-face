# api/main.py
from fastapi import FastAPI, UploadFile
from app.tasks.face_tasks import generate_face_task

app = FastAPI()


@app.post("/generate/")
async def generate(file: UploadFile):
    pass
    # # url = await save_to_minio(file)
    # task = generate_face_task.delay(url)
    # return {"task_id": task.id, "status_url": f"/status/{task.id}"}
