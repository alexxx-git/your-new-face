import numpy as np
import onnxruntime as ort
import cv2


class SCRFD:
    def __init__(
        self, det_path, input_size=(640, 640), score_threshold=0.5, nms_threshold=0.4
    ):
        self.detector = ort.InferenceSession(
            det_path, providers=["CPUExecutionProvider"]
        )
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.det_input = self.detector.get_inputs()[0].name
        self.strides = [8, 16, 32]

    def preprocess(self, image):
        img = cv2.resize(image, self.input_size)
        blob = img.astype(np.float32)
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)
        return blob

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

    def decode_outputs(self, outputs, scale_x, scale_y):
        """
        Декодирование выходов SCRFD модели в координаты оригинального изображения
        """
        num_heads = len(outputs) // 3
        scores_list = outputs[:num_heads]
        bboxes_list = outputs[num_heads : 2 * num_heads]

        proposals = []

        for stride, scores, bboxes in zip(self.strides, scores_list, bboxes_list):
            # Убираем лишние размерности
            scores = scores.squeeze(0)  # [1, H*W, 1] -> [H*W, 1]
            bboxes = bboxes.squeeze(0)  # [1, H*W, 4] -> [H*W, 4]

            if scores.ndim == 2:
                scores = scores[:, 0]  # [H*W, 1] -> [H*W]

            fm_h, fm_w = scores.shape[0], 1  # flat

            inds = np.where(scores > self.score_threshold)[0]

            for idx in inds:
                score = scores[idx]

                # Координаты сетки на фиче-карте
                grid_x = idx % (self.input_size[0] // stride)
                grid_y = idx // (self.input_size[0] // stride)

                # Центр анкера в пространстве модели (640×640)
                cx = (grid_x + 0.5) * stride
                cy = (grid_y + 0.5) * stride

                # Регрессия: расстояния от центра до границ (в пикселях модели)
                l, t, r, b = bboxes[idx]

                # 🔹 Декодим координаты в пространстве модели
                x1_model = cx - l
                y1_model = cy - t
                x2_model = cx + r
                y2_model = cy + b

                # 🔹 Масштабируем в координаты оригинального изображения
                x1 = x1_model * scale_x
                y1 = y1_model * scale_y
                x2 = x2_model * scale_x
                y2 = y2_model * scale_y

                proposals.append([x1, y1, x2, y2, score])

        if len(proposals) == 0:
            return None

        proposals = np.array(proposals)

        # NMS
        keep = self.nms(proposals)
        if len(keep) == 0:
            return None

        # Возвращаем лучший детекшн (или можно вернуть все: proposals[keep])
        best = proposals[keep[0]]
        return best[:4]  # [x1, y1, x2, y2]

    def detect(self, image):
        h, w, _ = image.shape
        blob = self.preprocess(image)
        scale_x = w / self.input_size[0]
        scale_y = h / self.input_size[1]

        outputs = self.detector.run(None, {self.det_input: blob})
        bbox = self.decode_outputs(outputs, scale_x, scale_y)
        return bbox

    def crop_face(self, image, bbox, scale=1.8):
        if bbox is None:
            return None
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


if __name__ == "__main__":
    scrfd = SCRFD("models/det_10g.onnx")
    img = cv2.imread("input1.jpg")
    assert img is not None, "Image not loaded!"

    bbox = scrfd.detect(img)
    face = scrfd.crop_face(img, bbox)

    print("bbox:", bbox)
    print("face shape:", None if face is None else face.shape)

    if face is None or face.size == 0:
        print("No face detected")
    else:
        cv2.imshow("face", face)
        cv2.waitKey(0)
