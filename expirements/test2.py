import cv2
import numpy as np
import onnxruntime as ort


class FaceEngine:
    def __init__(self, det_path, rec_path):
        self.detector = ort.InferenceSession(
            det_path, providers=["CPUExecutionProvider"]
        )
        self.recognizer = ort.InferenceSession(
            rec_path, providers=["CPUExecutionProvider"]
        )

        self.det_input = self.detector.get_inputs()[0].name
        self.rec_input = self.recognizer.get_inputs()[0].name

        self.input_size = (640, 640)
        self.nms_threshold = 0.4
        self.score_threshold = 0.5

    # -------------------------
    # DETECTION (SCRFD decode)
    # -------------------------
    def detect(self, image):
        h, w, _ = image.shape

        img = cv2.resize(image, self.input_size)
        scale_x = w / self.input_size[0]
        scale_y = h / self.input_size[1]

        blob = img.astype(np.float32)
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        outputs = self.detector.run(None, {self.det_input: blob})

        proposals = []

        strides = [8, 16, 32]
        num_heads = len(outputs) // 3

        scores_list = outputs[:num_heads]
        bboxes_list = outputs[num_heads : 2 * num_heads]

        for stride, scores, bboxes in zip(strides, scores_list, bboxes_list):
            scores = scores.reshape(-1)
            bboxes = bboxes.reshape(-1, 4)

            fm_w = self.input_size[0] // stride
            fm_h = self.input_size[1] // stride

            inds = np.where(scores > self.score_threshold)[0]

            for idx in inds:
                score = scores[idx]

                grid_x = idx % fm_w
                grid_y = idx // fm_w

                cx = grid_x * stride
                cy = grid_y * stride

                l, t, r, b = bboxes[idx]

                x1 = (cx - l) * scale_x
                y1 = (cy - t) * scale_y
                x2 = (cx + r) * scale_x
                y2 = (cy + b) * scale_y

                proposals.append([x1, y1, x2, y2, score])

        if len(proposals) == 0:
            return None

        proposals = np.array(proposals)
        keep = self.nms(proposals)

        best = proposals[keep[0]]
        return best[:4]

    # -------------------------
    # NMS
    # -------------------------
    def nms(self, boxes):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)

            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]

        return keep

    # -------------------------
    # CROP
    # -------------------------
    def crop_face(self, image, bbox, scale=1.8):
        h, w, _ = image.shape
        x1, y1, x2, y2 = bbox.astype(int)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        bw = int((x2 - x1) * scale)
        bh = int((y2 - y1) * scale)

        x1 = max(cx - bw // 2, 0)
        y1 = max(cy - bh // 2, 0)
        x2 = min(cx + bw // 2, w)
        y2 = min(cy + bh // 2, h)

        return image[y1:y2, x1:x2]

    # -------------------------
    # EMBEDDING (ArcFace)
    # -------------------------
    def get_embedding(self, face_img):
        face = cv2.resize(face_img, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5

        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)

        embedding = self.recognizer.run(None, {self.rec_input: face})[0]
        return embedding

    # -------------------------
    # FULL PIPELINE
    # -------------------------
    def process(self, image):
        bbox = self.detect(image)

        if bbox is None:
            raise Exception("Face not found")

        face = self.crop_face(image, bbox)
        emb = self.get_embedding(face)
        print("bbox:", bbox)
        return {"bbox": bbox, "face": face, "embedding": emb}


engine = FaceEngine(
    "models/det_10g.onnx",
    "models/w600k_r50.onnx",
)

img = cv2.imread("input.jpg")

result = engine.process(img)

print("Embedding shape:", result["embedding"].shape)

cv2.imshow("face", result["face"])
cv2.waitKey(0)
