import cv2
import numpy as np
import torch
from PIL import Image
from insightface.app import FaceAnalysis
from typing import Tuple, Optional
import os
from datetime import datetime


class AgeTransformationPipeline:
    """
    ML пайплайн для трансформации возраста на фото
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.face_app = FaceAnalysis(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # Загрузка моделей (пример для StyleGAN + Age Condition)
        self.age_generator = self._load_age_generator()
        self.face_enhancer = self._load_face_enhancer()

    def _load_age_generator(self):
        """
        Загрузка модели генерации возраста
        В продакшене использовать предобученную модель
        """
        # Пример: StyleGAN2 с conditioning по возрасту
        # Для демо используем заглушку
        model = torch.hub.load(
            "rosinality/stylegan2-pytorch",
            "generator",
            "stylegan2-ffhq-config-f.pt",
            channel_multiplier=2,
        )
        model.to(self.device)
        model.eval()
        return model

    def _load_face_enhancer(self):
        """
        Загрузка модели улучшения качества лица (GFPGAN)
        """
        try:
            from gfpgan import GFPGANer

            enhancer = GFPGANer(
                model_path="experiments/pretrained_models/GFPGANv1.4.pth",
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            return enhancer
        except:
            print("GFPGAN не доступен, используем базовое улучшение")
            return None

    def detect_and_align_face(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Детекция и выравнивание лица

        Args:
            image: Input изображение в формате BGR

        Returns:
            aligned_face: Выровненное лицо
            face_info: Информация о лице (bbox, landmarks, age, gender)
        """
        faces = self.face_app.get(image)

        if len(faces) == 0:
            return None, None

        # Берём первое обнаруженное лицо (можно добавить выбор крупнейшего)
        face = faces[0]

        # Извлекаем bounding box
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        # Добавляем отступы для контекста
        margin = int((x2 - x1) * 0.3)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)

        # Кроп лица
        face_crop = image[y1:y2, x1:x2]

        # Выравнивание по landmarks
        landmarks = face.kps
        aligned_face = self._align_face(face_crop, landmarks)

        face_info = {
            "bbox": bbox,
            "landmarks": landmarks,
            "detected_age": face.age,
            "gender": face.sex,
            "crop_coords": (x1, y1, x2, y2),
            "original_size": image.shape[:2],
        }

        return aligned_face, face_info

    def _align_face(
        self,
        face: np.ndarray,
        landmarks: np.ndarray,
        output_size: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """
        Выравнивание лица по ключевым точкам

        Args:
            face: Кроп лица
            landmarks: 5 ключевых точек (глаза, нос, уголки рта)
            output_size: Размер выходного изображения

        Returns:
            aligned_face: Выровненное изображение
        """
        # Стандартные точки для выравнивания (FFHQ стиль)
        src_pts = np.array(
            [
                [192.98138, 239.94708],
                [318.90277, 240.1936],
                [256.63416, 314.01935],
                [201.26117, 371.41043],
                [313.08905, 371.15118],
            ],
            dtype=np.float32,
        )

        # Масштабируем под наш output_size
        scale = output_size[0] / 512.0
        src_pts *= scale

        # Вычисляем матрицу аффинного преобразования
        M, _ = cv2.estimateAffinePartial2D(landmarks, src_pts)

        # Применяем трансформацию
        aligned = cv2.warpAffine(face, M, output_size, borderMode=cv2.BORDER_REFLECT)

        return aligned

    def transform_age(
        self, face_image: np.ndarray, target_age: int, current_age: Optional[int] = None
    ) -> np.ndarray:
        """
        Трансформация возраста лица

        Args:
            face_image: Выровненное лицо (RGB, 512x512)
            target_age: Целевой возраст
            current_age: Текущий возраст (если известен)

        Returns:
            transformed_face: Лицо с изменённым возрастом
        """
        # Конвертация в тензор
        face_tensor = self._preprocess_face(face_image)

        # Вычисление латентного вектора
        with torch.no_grad():
            latent = self.age_generator.style([1])

            # Age conditioning (вектор возраста)
            age_delta = target_age - (current_age or 25)
            age_vector = self._encode_age(age_delta)

            # Модификация латентного пространства
            modified_latent = self._apply_age_conditioning(latent, age_vector)

            # Генерация изображения
            generated, _ = self.age_generator(
                [modified_latent], input_is_latent=True, randomize_noise=False
            )

        # Пост-обработка
        output = self._postprocess_face(generated, face_image)

        return output

    def _preprocess_face(self, face: np.ndarray) -> torch.Tensor:
        """
        Препроцессинг изображения для модели
        """
        # Конвертация BGR -> RGB
        if len(face.shape) == 3 and face.shape[2] == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Нормализация [-1, 1]
        face = face.astype(np.float32) / 127.5 - 1.0

        # HWC -> CHW
        face = np.transpose(face, (2, 0, 1))

        # Добавляем batch dimension
        face = np.expand_dims(face, 0)

        return torch.from_numpy(face).to(self.device)

    def _encode_age(self, age_delta: int) -> torch.Tensor:
        """
        Кодирование возраста в вектор для conditioning

        В продакшене использовать обученный encoder
        """
        # Нормализация возраста [-1, 1]
        age_normalized = np.clip(age_delta / 50.0, -1.0, 1.0)

        # Создаём вектор (размер зависит от архитектуры модели)
        age_vector = torch.FloatTensor([age_normalized] * 512).to(self.device)

        return age_vector

    def _apply_age_conditioning(
        self, latent: torch.Tensor, age_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Применение age conditioning к латентному вектору

        Это упрощённая версия - в продакшене использовать обученный модуль
        """
        # Простая линейная интерполяция в латентном пространстве
        # В реальности нужна обученная модель для точного контроля возраста
        alpha = torch.sigmoid(age_vector[:1].mean()) * 0.3

        # Смешивание с референсным латентом нужного возраста
        modified = latent + alpha * age_vector.unsqueeze(0).unsqueeze(0)

        return modified

    def _postprocess_face(
        self, generated: torch.Tensor, original: np.ndarray
    ) -> np.ndarray:
        """
        Пост-обработка сгенерированного лица
        """
        # Конвертация тензора в изображение
        img = generated.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = np.clip((img + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)

        # Улучшение качества через GFPGAN
        if self.face_enhancer is not None:
            _, _, img = self.face_enhancer.enhance(img, has_aligned=True)

        # RGB -> BGR для OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def blend_with_original(
        self, transformed_face: np.ndarray, original_image: np.ndarray, face_info: dict
    ) -> np.ndarray:
        """
        Блендинг трансформированного лица с оригинальным изображением

        Args:
            transformed_face: Трансформированное лицо
            original_image: Оригинальное изображение
            face_info: Информация о лице из детекции

        Returns:
            blended_image: Итоговое изображение
        """
        x1, y1, x2, y2 = face_info["crop_coords"]

        # Ресайз трансформированного лица под кроп
        transformed_resized = cv2.resize(transformed_face, (x2 - x1, y2 - y1))

        # Создаём маску для плавного блендинга
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.ellipse(
            mask,
            ((x1 + x2) // 2, (y1 + y2) // 2),
            ((x2 - x1) // 2, (y2 - y1) // 2),
            0,
            0,
            360,
            255,
            -1,
        )

        # Гауссово размытие маски для плавных краёв
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = mask.astype(np.float32) / 255.0

        # Блендинг
        if len(mask.shape) == 2:
            mask = np.stack([mask] * 3, axis=2)

        blended = (
            transformed_resized * mask + original_image[y1:y2, x1:x2] * (1 - mask)
        ).astype(np.uint8)

        # Вставка обратно в оригинальное изображение
        result = original_image.copy()
        result[y1:y2, x1:x2] = blended

        return result

    def process(
        self, input_path: str, target_age: int, output_path: Optional[str] = None
    ) -> dict:
        """
        Полный пайплайн обработки

        Args:
            input_path: Путь к входному изображению
            target_age: Целевой возраст
            output_path: Путь для сохранения результата

        Returns:
            result: Словарь с результатом и метаданными
        """
        start_time = datetime.now()

        # 1. Загрузка изображения
        original_image = cv2.imread(input_path)
        if original_image is None:
            raise ValueError(f"Не удалось загрузить изображение: {input_path}")

        # 2. Детекция и выравнивание лица
        aligned_face, face_info = self.detect_and_align_face(original_image)
        if aligned_face is None:
            raise ValueError("Лицо не обнаружено на изображении")

        # 3. Трансформация возраста
        transformed_face = self.transform_age(
            aligned_face, target_age, face_info.get("detected_age")
        )

        # 4. Блендинг с оригиналом
        result_image = self.blend_with_original(
            transformed_face, original_image, face_info
        )

        # 5. Сохранение результата
        if output_path:
            cv2.imwrite(output_path, result_image)

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "output_path": output_path,
            "detected_age": face_info.get("detected_age"),
            "target_age": target_age,
            "gender": face_info.get("gender"),
            "processing_time": processing_time,
            "face_bbox": face_info["bbox"].tolist(),
        }


# ==================== Пример использования ====================

if __name__ == "__main__":
    # Инициализация пайплайна
    pipeline = AgeTransformationPipeline(device="cuda")

    # Параметры
    input_image = "input.jpg"
    target_age = 50  # Целевой возраст
    output_image = "output_age_50.jpg"

    try:
        # Запуск обработки
        result = pipeline.process(
            input_path=input_image, target_age=target_age, output_path=output_image
        )

        print(f"✅ Обработка завершена успешно!")
        print(f"📊 detected_age: {result['detected_age']}")
        print(f"🎯 target_age: {result['target_age']}")
        print(f"⏱️ processing_time: {result['processing_time']:.2f}s")
        print(f"📁 output: {result['output_path']}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
