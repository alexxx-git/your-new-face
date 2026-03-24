import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort

app = FaceAnalysis(
    name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))
# инициализация
# app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
# app.prepare(ctx_id=0, det_size=(640, 640))  # 640x640 — стандарт

# # загрузка изображения
# img = cv2.imread("input1.jpg")

# # детекция
# faces = app.get(img)

# print("Faces found:", len(faces))

# for face in faces:
#     print("bbox:", face.bbox)
#     print("kps:", face.kps)
#     print("score:", face.det_score)

#     # рисуем bbox
#     bbox = face.bbox.astype(int)
#     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

# cv2.imwrite("result.jpg", img)
