import cv2
import numpy as np
import onnxruntime as ort


# --- utils ---
def preprocess(img, size=(640, 640)):
    img_resized = cv2.resize(img, size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = (img_rgb - 127.5) / 128.0
    img_transposed = np.transpose(img_norm, (2, 0, 1)).astype(np.float32)
    return np.expand_dims(img_transposed, axis=0), img_resized


# --- load models ---
det_sess = ort.InferenceSession("models/det_10g.onnx")
rec_sess = ort.InferenceSession("models/w600k_r50.onnx")

# --- load image ---
img = cv2.imread("input.jpg")
input_tensor, resized = preprocess(img)

# --- detection ---

outputs = det_sess.run(None, {"input.1": input_tensor})

for i, out in enumerate(outputs):
    print(i, out.shape)
# ⚠️ здесь зависит от модели — упрощённо:
bboxes = outputs[0]  # (N, 5)
landmarks = outputs[1]  # (N, 10)

# берём первое лицо
bbox = bboxes[0][:4].astype(int)
print(bboxes)
# --- crop face ---
x1, y1, x2, y2 = bbox
face = resized[y1:y2, x1:x2]

# --- preprocess face for embedding ---
face = cv2.resize(face, (112, 112))
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
face = np.transpose(face, (2, 0, 1)).astype(np.float32)
face = (face - 127.5) / 128.0
face = np.expand_dims(face, axis=0)

# --- embedding ---
embedding = rec_sess.run(None, {"data": face})[0]

print("Embedding shape:", embedding.shape)

# --- draw bbox ---
cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite("output.jpg", resized)
