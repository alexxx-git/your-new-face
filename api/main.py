from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.celery_app import app as celery_app
from workers.tasks import process_face

app = FastAPI(
    title="Your New Face API", docs_url="/api/docs", openapi_url="/api/openapi.json"
)

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


@app.get("/api/health")
async def health():
    return {"status": "ok", "celery": "ready"}


@app.post("/api/tasks")
async def create_task(request: TaskRequest):
    task = process_face.delay(request.dict())
    return {"task_id": task.id, "status": "PENDING"}


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }
