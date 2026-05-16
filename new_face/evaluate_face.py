"""
evaluate_face.py

Оценка качества сгенерированного лица по метрикам:
  1. Cosine Similarity (идентичность, ArcFace via InsightFace)
  2. Age Estimation MAE (точность возраста, InsightFace)
  3. BRISQUE (качество без референса)
  4. LPIPS (перцептивное сходство)

Все модели метрик запускаются на CPU, чтобы не занимать VRAM.

Использование:
  python evaluate_face.py --original path/to/original.jpg \
                          --generated path/to/generated.jpg \
                          --target_age 70
"""

import argparse
import json
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(description="Face generation metrics")
    parser.add_argument("--original", required=True, help="Путь к исходному фото")
    parser.add_argument(
        "--generated", required=True, help="Путь к сгенерированному фото"
    )
    parser.add_argument(
        "--target_age",
        required=True,
        type=int,
        help="Целевой возраст (введён пользователем)",
    )
    parser.add_argument(
        "--det_size",
        default=640,
        type=int,
        help="Размер детектора лиц (по умолчанию 640)",
    )
    parser.add_argument(
        "--skip_brisque",
        action="store_true",
        help="Не считать BRISQUE",
    )
    parser.add_argument(
        "--skip_lpips",
        action="store_true",
        help="Не считать LPIPS",
    )
    parser.add_argument(
        "--debug_dir",
        default=None,
        help="Папка для сохранения debug-crop/aligned изображений и JSON-отчёта",
    )
    return parser.parse_args()


@lru_cache(maxsize=2)
def get_face_app(det_size: int):
    """Инициализирует InsightFace на CPU и переиспользует модель между вызовами."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def get_main_face(app, img_bgr: np.ndarray, label: str):
    faces = app.get(img_bgr)
    if len(faces) == 0:
        print(f"[WARN] Лицо не найдено на изображении: {label}")
        return None
    if len(faces) > 1:
        print(f"[INFO] Найдено {len(faces)} лиц на '{label}', берём наибольшее")
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def load_rgb_image(img_path: str) -> Image.Image:
    """Открывает изображение с учётом EXIF orientation и приводит к RGB."""
    return ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")


def load_bgr_image(img_path: str) -> np.ndarray:
    rgb = load_rgb_image(img_path)
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def get_face_embedding(face) -> np.ndarray | None:
    return getattr(face, "normed_embedding", None)


def get_face_age(face) -> float | None:
    age = getattr(face, "age", None)
    if age is None:
        return None
    return float(age)


def get_face_score(face) -> float | None:
    score = getattr(face, "det_score", None)
    if score is None:
        return None
    return float(score)


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    return float(np.dot(emb_a, emb_b))


def compute_brisque(img_path: str) -> float | None:
    """
    BRISQUE: меньше = лучше (0-100, идеал около 0, плохое качество около 60+).
    """
    try:
        from brisque import BRISQUE

        brisque_obj = BRISQUE(url=False)
        rgb_image = load_rgb_image(img_path)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp_file:
            rgb_image.save(tmp_file.name, format="JPEG", quality=95)
            score = brisque_obj.score(img=tmp_file.name)
        return float(score)
    except Exception as e:
        print(f"[WARN] BRISQUE failed: {e}")
        return None


@lru_cache(maxsize=1)
def get_lpips_model():
    import lpips

    return lpips.LPIPS(net="alex", verbose=False).cpu()


def compute_lpips(img_path_a: str, img_path_b: str) -> float | None:
    """
    LPIPS: меньше = более похожи перцептивно (0 = идентичны).
    Считается на CPU.
    """
    try:
        image_a = load_rgb_image(img_path_a).resize((256, 256))
        image_b = load_rgb_image(img_path_b).resize((256, 256))
        return compute_lpips_images(image_a, image_b)
    except Exception as e:
        print(f"[WARN] LPIPS failed: {e}")
        return None


def compute_lpips_images(image_a: Image.Image, image_b: Image.Image) -> float | None:
    try:
        import torch
        from torchvision import transforms

        loss_fn = get_lpips_model()

        def to_tensor(image: Image.Image):
            tensor = transforms.ToTensor()(image.convert("RGB").resize((256, 256)))
            tensor = tensor * 2 - 1
            return tensor.unsqueeze(0).cpu()

        tensor_a = to_tensor(image_a)
        tensor_b = to_tensor(image_b)

        with torch.no_grad():
            score = loss_fn(tensor_a, tensor_b)
        return float(score.item())
    except Exception as e:
        print(f"[WARN] LPIPS failed: {e}")
        return None


def get_aligned_face_image(
    image_rgb: Image.Image,
    face,
    output_size: int = 256,
) -> Image.Image:
    """
    Выравнивает лицо по 5 landmarks из InsightFace.
    Если landmarks недоступны, использует crop по bbox с padding.
    """
    image_np = np.array(image_rgb)
    keypoints = getattr(face, "kps", None)
    if keypoints is not None and len(keypoints) >= 5:
        template = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        template *= output_size / 112.0
        matrix, _ = cv2.estimateAffinePartial2D(
            np.asarray(keypoints, dtype=np.float32),
            template,
            method=cv2.LMEDS,
        )
        if matrix is not None:
            aligned = cv2.warpAffine(
                image_np,
                matrix,
                (output_size, output_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            return Image.fromarray(aligned)

    return crop_face_image(image_rgb, face, output_size=output_size)


def crop_face_image(
    image_rgb: Image.Image,
    face,
    output_size: int = 256,
    padding_ratio: float = 0.25,
) -> Image.Image:
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return image_rgb.resize((output_size, output_size))

    width, height = image_rgb.size
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_width = x2 - x1
    box_height = y2 - y1
    padding = max(box_width, box_height) * padding_ratio

    left = max(0, int(x1 - padding))
    top = max(0, int(y1 - padding))
    right = min(width, int(x2 + padding))
    bottom = min(height, int(y2 + padding))

    if right <= left or bottom <= top:
        return image_rgb.resize((output_size, output_size))

    return image_rgb.crop((left, top, right, bottom)).resize((output_size, output_size))


def mirror_rgb_image(image_rgb: Image.Image) -> Image.Image:
    return ImageOps.mirror(image_rgb)


def face_to_debug_dict(face) -> dict[str, Any] | None:
    if face is None:
        return None

    bbox = getattr(face, "bbox", None)
    keypoints = getattr(face, "kps", None)
    return {
        "bbox": bbox.tolist() if bbox is not None else None,
        "keypoints": keypoints.tolist() if keypoints is not None else None,
        "det_score": get_face_score(face),
        "age": get_face_age(face),
    }


def save_debug_artifacts(
    debug_dir: str,
    image_orig_rgb: Image.Image,
    image_gen_rgb: Image.Image,
    face_orig,
    face_gen,
    face_gen_mirrored,
    metrics: dict[str, Any],
) -> None:
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)

    image_orig_rgb.save(debug_path / "original_rgb.jpg", quality=95)
    image_gen_rgb.save(debug_path / "generated_rgb.jpg", quality=95)
    mirror_rgb_image(image_gen_rgb).save(debug_path / "generated_mirrored_rgb.jpg", quality=95)

    if face_orig is not None:
        get_aligned_face_image(image_orig_rgb, face_orig).save(
            debug_path / "original_aligned.jpg", quality=95
        )
        crop_face_image(image_orig_rgb, face_orig).save(
            debug_path / "original_crop.jpg", quality=95
        )

    if face_gen is not None:
        get_aligned_face_image(image_gen_rgb, face_gen).save(
            debug_path / "generated_aligned.jpg", quality=95
        )
        crop_face_image(image_gen_rgb, face_gen).save(
            debug_path / "generated_crop.jpg", quality=95
        )

    if face_gen_mirrored is not None:
        mirrored_image = mirror_rgb_image(image_gen_rgb)
        get_aligned_face_image(mirrored_image, face_gen_mirrored).save(
            debug_path / "generated_mirrored_aligned.jpg", quality=95
        )
        crop_face_image(mirrored_image, face_gen_mirrored).save(
            debug_path / "generated_mirrored_crop.jpg", quality=95
        )

    with (debug_path / "metrics_debug.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def compute_composite(
    cos_sim: float | None,
    age_mae: float | None,
    brisque: float | None,
    lpips_score: float | None,
) -> float | None:
    """
    Нормализуем каждую метрику в [0, 1] (1 = лучше) и берём взвешенное среднее.
    """
    scores = {}
    weights = {}

    if cos_sim is not None:
        scores["identity"] = (cos_sim + 1) / 2
        weights["identity"] = 0.45

    if age_mae is not None:
        scores["age"] = max(0.0, 1.0 - age_mae / 30.0)
        weights["age"] = 0.30

    if brisque is not None:
        scores["brisque"] = max(0.0, 1.0 - brisque / 100.0)
        weights["brisque"] = 0.10

    if lpips_score is not None:
        scores["lpips"] = max(0.0, 1.0 - lpips_score)
        weights["lpips"] = 0.15

    if not scores:
        return None

    total_weight = sum(weights[key] for key in scores)
    composite = sum(scores[key] * weights[key] for key in scores) / total_weight
    return round(composite, 4)


def evaluate_face(
    original_path: str,
    generated_path: str,
    target_age: int,
    det_size: int = 640,
    with_brisque: bool = True,
    with_lpips: bool = True,
    debug_dir: str | None = None,
) -> dict[str, Any]:
    """
    Возвращает словарь метрик. Эту функцию можно вызывать из worker-а.
    """
    try:
        image_orig_rgb = load_rgb_image(original_path)
        image_gen_rgb = load_rgb_image(generated_path)
    except Exception as e:
        raise ValueError(f"Не удалось загрузить изображения: {e}") from e

    img_orig = cv2.cvtColor(np.array(image_orig_rgb), cv2.COLOR_RGB2BGR)
    img_gen = cv2.cvtColor(np.array(image_gen_rgb), cv2.COLOR_RGB2BGR)

    if img_orig is None:
        raise ValueError(f"Не удалось загрузить: {original_path}")
    if img_gen is None:
        raise ValueError(f"Не удалось загрузить: {generated_path}")

    cos_sim = None
    predicted_age = None
    age_mae = None
    face_orig = None
    face_gen = None
    face_gen_mirrored = None
    mirrored_cos_sim = None

    try:
        app = get_face_app(det_size)
        face_orig = get_main_face(app, img_orig, "original")
        face_gen = get_main_face(app, img_gen, "generated")

        if face_orig is not None and face_gen is not None:
            emb_orig = get_face_embedding(face_orig)
            emb_gen = get_face_embedding(face_gen)
            if emb_orig is not None and emb_gen is not None:
                cos_sim = cosine_similarity(emb_orig, emb_gen)

            predicted_age = get_face_age(face_gen)
            if predicted_age is not None:
                age_mae = abs(predicted_age - target_age)

        mirrored_gen_rgb = mirror_rgb_image(image_gen_rgb)
        mirrored_gen_bgr = cv2.cvtColor(np.array(mirrored_gen_rgb), cv2.COLOR_RGB2BGR)
        face_gen_mirrored = get_main_face(app, mirrored_gen_bgr, "generated_mirrored")
        if face_orig is not None and face_gen_mirrored is not None:
            emb_orig = get_face_embedding(face_orig)
            emb_gen_mirrored = get_face_embedding(face_gen_mirrored)
            if emb_orig is not None and emb_gen_mirrored is not None:
                mirrored_cos_sim = cosine_similarity(emb_orig, emb_gen_mirrored)
    except ImportError:
        print("[WARN] insightface не установлен. Пропускаем identity/age metrics.")
    except Exception as e:
        print(f"[WARN] InsightFace metrics failed: {e}")

    brisque_score = compute_brisque(generated_path) if with_brisque else None
    lpips_score = None
    if with_lpips:
        if face_orig is not None and face_gen is not None:
            aligned_orig = get_aligned_face_image(image_orig_rgb, face_orig)
            aligned_gen = get_aligned_face_image(image_gen_rgb, face_gen)
            lpips_score = compute_lpips_images(aligned_orig, aligned_gen)
        else:
            print("[WARN] Face alignment unavailable. LPIPS uses full resized images.")
            lpips_score = compute_lpips(original_path, generated_path)

    composite = compute_composite(cos_sim, age_mae, brisque_score, lpips_score)

    metrics = {
        "cosine_similarity": cos_sim,
        "cosine_similarity_mirrored_generated": mirrored_cos_sim,
        "predicted_age": predicted_age,
        "target_age": target_age,
        "age_mae": age_mae,
        "brisque": brisque_score,
        "lpips": lpips_score,
        "composite_score": composite,
        "metrics_json": {
            "original_path": str(Path(original_path)),
            "generated_path": str(Path(generated_path)),
            "det_size": det_size,
            "with_brisque": with_brisque,
            "with_lpips": with_lpips,
            "original_face": face_to_debug_dict(face_orig),
            "generated_face": face_to_debug_dict(face_gen),
            "generated_mirrored_face": face_to_debug_dict(face_gen_mirrored),
        },
    }

    if debug_dir:
        save_debug_artifacts(
            debug_dir=debug_dir,
            image_orig_rgb=image_orig_rgb,
            image_gen_rgb=image_gen_rgb,
            face_orig=face_orig,
            face_gen=face_gen,
            face_gen_mirrored=face_gen_mirrored,
            metrics=metrics,
        )

    return metrics


def print_results(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 55)
    print("  РЕЗУЛЬТАТЫ")
    print("=" * 55)

    cos_sim = metrics["cosine_similarity"]
    if cos_sim is not None:
        verdict = (
            "хорошо" if cos_sim >= 0.75 else ("слабо" if cos_sim >= 0.5 else "плохо")
        )
        print(f"  Cosine Similarity (идентичность) : {cos_sim:.4f}  {verdict}")
    else:
        print("  Cosine Similarity                : N/A")

    mirrored_cos_sim = metrics.get("cosine_similarity_mirrored_generated")
    if mirrored_cos_sim is not None:
        verdict = (
            "хорошо"
            if mirrored_cos_sim >= 0.75
            else ("слабо" if mirrored_cos_sim >= 0.5 else "плохо")
        )
        print(
            f"  Cosine Similarity (mirrored gen) : {mirrored_cos_sim:.4f}  {verdict}"
        )

    predicted_age = metrics["predicted_age"]
    age_mae = metrics["age_mae"]
    if predicted_age is not None and age_mae is not None:
        print(f"  Predicted age                    : {predicted_age:.1f} лет")
        print(f"  Target age                       : {metrics['target_age']} лет")
        verdict = (
            "хорошо" if age_mae <= 5 else ("допустимо" if age_mae <= 10 else "плохо")
        )
        print(f"  Age MAE                          : {age_mae:.1f} лет  {verdict}")
    else:
        print("  Age MAE                          : N/A")

    brisque_score = metrics["brisque"]
    if brisque_score is not None:
        verdict = (
            "хорошо"
            if brisque_score <= 30
            else ("допустимо" if brisque_score <= 50 else "плохо")
        )
        print(f"  BRISQUE (lower is better)        : {brisque_score:.2f}  {verdict}")
    else:
        print("  BRISQUE                          : N/A")

    lpips_score = metrics["lpips"]
    if lpips_score is not None:
        verdict = (
            "хорошо"
            if lpips_score <= 0.3
            else ("допустимо" if lpips_score <= 0.5 else "плохо")
        )
        print(f"  LPIPS (lower is better)          : {lpips_score:.4f}  {verdict}")
    else:
        print("  LPIPS                            : N/A")

    print("-" * 55)
    composite = metrics["composite_score"]
    if composite is not None:
        bar_len = int(composite * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        print(f"  Composite Score (higher is better): {composite:.4f}")
        print(f"  [{bar}]")
    else:
        print("  Composite Score                  : N/A")
    print("=" * 55 + "\n")


def main():
    args = parse_args()

    print("\n" + "=" * 55)
    print("  Face Generation Evaluation")
    print("=" * 55)
    print(f"  Original  : {args.original}")
    print(f"  Generated : {args.generated}")
    print(f"  Target age: {args.target_age}")
    print("  Device    : CPU")
    print("=" * 55 + "\n")

    try:
        metrics = evaluate_face(
            original_path=args.original,
            generated_path=args.generated,
            target_age=args.target_age,
            det_size=args.det_size,
            with_brisque=not args.skip_brisque,
            with_lpips=not args.skip_lpips,
            debug_dir=args.debug_dir,
        )
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")

    print_results(metrics)


if __name__ == "__main__":
    main()
