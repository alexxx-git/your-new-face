"""
evaluate_face.py

Оценка качества сгенерированного лица по метрикам:
  1. Cosine Similarity (идентичность, ArcFace via InsightFace) — на выровненных
     кропах одинакового размера (см. IDENTITY_ALIGN_SIZE).
  2. Age Estimation MAE (точность возраста, InsightFace)
  3. LPIPS (перцептивное сходство на aligned-crop)

Все модели метрик запускаются на CPU, чтобы не занимать VRAM.

Использование:
  python evaluate_face.py --original path/to/original.jpg \\
                          --generated path/to/generated.jpg \\
                          --target_age 70

Опционально: ``--debug_dir`` — сохранить отладочные кадры и JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

# Размер выравнивания для ArcFace-эмбеддинга: original и generated приводятся
# к одному каноническому виду (affine по 5 точкам или fallback crop).
IDENTITY_ALIGN_SIZE = 112

ALIGNMENT_NOTE = (
    "Выравнивание лица: если у детектора есть 5 ключевых точек — "
    f"ArcFace norm_crop или affine по тому же шаблону ({IDENTITY_ALIGN_SIZE}×{IDENTITY_ALIGN_SIZE}); "
    "если точек нет — кроп по bounding box (bbox) с padding и resize. "
    "LPIPS считается на aligned 256×256 по тем же правилам (или bbox, если точек нет)."
)

_REPO_ROOT = Path(__file__).resolve().parent
_INSIGHTFACE_ROOT = os.getenv(
    "YNF_INSIGHTFACE_ROOT", str(_REPO_ROOT)
)


def resolve_image_path(path_str: str, label: str) -> Path:
    """
    Абсолютный путь к существующему файлу. Пробует: как есть, смена расширения
    (.jpg/.png/.jpeg/.webp), cwd, корень репозитория.
    """
    raw = Path(path_str).expanduser()

    def extension_variants(p: Path) -> list[Path]:
        if not p.suffix:
            return [p]
        base = p.with_suffix("")
        alts = [p]
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG"):
            if ext.lower() != p.suffix.lower():
                q = base.with_suffix(ext)
                if q not in alts:
                    alts.append(q)
        return alts

    candidates: list[Path] = []
    for variant in extension_variants(raw):
        if variant.is_absolute():
            candidates.append(variant)
        else:
            candidates.append(Path.cwd() / variant)
            candidates.append(_REPO_ROOT / variant)
            candidates.append(_REPO_ROOT / variant.name)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    tried = "\n  ".join(str(c) for c in candidates[:24])
    more = "" if len(candidates) <= 24 else f"\n  ... всего вариантов: {len(candidates)}"
    raise FileNotFoundError(
        f"Файл не найден ({label}): {path_str!r}\n"
        f"Проверено (часть):\n  {tried}{more}\n"
        f"Текущая папка: {Path.cwd()}"
    )


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


@lru_cache(maxsize=16)
def get_face_app(det_size: int):
    """Инициализирует InsightFace на CPU и переиспользует модель между вызовами."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        root=_INSIGHTFACE_ROOT,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def _det_size_chain(preferred: int) -> list[int]:
    """Убывающие размеры детектора: мелкие лица / сильный арт иногда ловятся только на меньшем det."""
    chain = [preferred, 640, 512, 480, 416, 384, 320, 256, 224, 192, 160, 128]
    out: list[int] = []
    for x in chain:
        x = int(x)
        if x >= 32 and x not in out:
            out.append(x)
    return out


def get_main_face_multiscale(
    img_bgr: np.ndarray,
    label: str,
    preferred_det: int = 512,
):
    """
    Ищет главное лицо, перебирая det_size. Возвращает (face | None, det_used | None).
    """
    for ds in _det_size_chain(preferred_det):
        app = get_face_app(ds)
        faces = app.get(img_bgr)
        if not faces:
            continue
        best = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        if len(faces) > 1:
            print(
                f"[INFO] На «{label}» найдено {len(faces)} лиц (det_size={ds}), берём наибольшее"
            )
        elif ds != preferred_det:
            print(
                f"[INFO] На «{label}» лицо найдено с det_size={ds} "
                f"(с {preferred_det}×{preferred_det} детектор не сработал)"
            )
        return best, ds

    h, w = img_bgr.shape[:2]
    print(
        f"[WARN] Детектор не нашёл лицо на «{label}» (изображение загружено, {w}×{h} px). "
        "Это не «файл не найден»: часто из‑за артефактов генерации, профиля, слишком мелкого лица "
        f"или нестандартного кадра. Перебор det_size: {_det_size_chain(preferred_det)}"
    )
    return None, None


def load_rgb_image(img_path: str) -> Image.Image:
    """Открывает изображение с учётом EXIF orientation и приводит к RGB."""
    return ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")


def get_face_embedding(face) -> np.ndarray | None:
    return getattr(face, "normed_embedding", None)


def get_face_age(face) -> float | None:
    age = getattr(face, "age", None)
    if age is None:
        return None
    return float(age)


def estimate_insightface_source_age(
    image_path: str,
    *,
    det_size: int = 640,
) -> tuple[float | None, dict[str, Any]]:
    """
    Возраст лица на изображении: InsightFace buffalo_l + тот же multiscale-детект,
    что и в ``evaluate_face`` (без генерации, только входной кадр).

    Возвращает (возраст_лет | None, meta с det_used и path_resolved).
    """
    path = resolve_image_path(image_path, "source_age")
    rgb = load_rgb_image(str(path))
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    face, det_used = get_main_face_multiscale(
        bgr, "source_age", preferred_det=int(det_size)
    )
    meta: dict[str, Any] = {"det_used": det_used, "path_resolved": str(path)}
    if face is None:
        return None, meta
    age = get_face_age(face)
    if age is None:
        return None, meta
    return float(age), meta


def get_face_score(face) -> float | None:
    score = getattr(face, "det_score", None)
    if score is None:
        return None
    return float(score)


def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    v = emb.astype(np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return emb.astype(np.float32)
    return (v / n).astype(np.float32)


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    a = _l2_normalize(emb_a)
    b = _l2_normalize(emb_b)
    return float(np.clip(np.dot(a.astype(np.float64), b.astype(np.float64)), -1.0, 1.0))


def arcface_template(output_size: int) -> np.ndarray:
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
    return template * (float(output_size) / 112.0)


def warp_face_by_landmarks(
    image_np: np.ndarray,
    keypoints,
    output_size: int,
) -> np.ndarray | None:
    matrix, _ = cv2.estimateAffinePartial2D(
        np.asarray(keypoints, dtype=np.float32),
        arcface_template(output_size),
        method=cv2.LMEDS,
    )
    if matrix is None:
        return None
    return cv2.warpAffine(
        image_np,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def aligned_face_bgr_for_metrics(
    img_bgr: np.ndarray,
    face,
    align_size: int = IDENTITY_ALIGN_SIZE,
) -> np.ndarray:

    if face is None:
        raise ValueError("face is None")

    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) >= 5:
        aligned = warp_face_by_landmarks(img_bgr, kps, align_size)
        if aligned is not None:
            return aligned

    rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    aligned_rgb = get_aligned_face_image(rgb, face, output_size=align_size)
    return cv2.cvtColor(np.array(aligned_rgb), cv2.COLOR_RGB2BGR)


def _recognition_feat_from_aligned_bgr(app, aimg_bgr: np.ndarray) -> np.ndarray | None:

    rec = getattr(app, "models", None)
    if not isinstance(rec, dict):
        rec = {}
    rec_model = rec.get("recognition")
    if rec_model is None:
        return None
    for arg in (aimg_bgr, [aimg_bgr]):
        try:
            feat = rec_model.get_feat(arg)
            arr = np.asarray(feat, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                return arr
        except Exception:
            continue
    return None


def compute_metric_embedding(
    app,
    img_bgr: np.ndarray,
    face,
    label: str,
    align_size: int = IDENTITY_ALIGN_SIZE,
) -> tuple[np.ndarray | None, bool]:
    """
    Эмбеддинг для косинуса: aligned align_size×align_size, затем вектор с головы
    recognition (ArcFace) на этом кропе — без повторного детекта на 112×112.

    Если recognition недоступен — апскейл кропа и app.get.
    Второй элемент кортежа — True, если использован embedding с полного кадра.
    """
    if face is None:
        return None, False

    try:
        aimg = aligned_face_bgr_for_metrics(img_bgr, face, align_size=align_size)
    except Exception as exc:
        print(f"[WARN] aligned_face_bgr_for_metrics ({label}): {exc}")
        emb = get_face_embedding(face)
        return (_l2_normalize(emb) if emb is not None else None, True)

    feat_rec = _recognition_feat_from_aligned_bgr(app, aimg)
    if feat_rec is not None:
        return _l2_normalize(feat_rec), False

    h, w = aimg.shape[:2]
    side = max(h, w)
    min_side_for_det = 320
    if side < min_side_for_det:
        scale = float(min_side_for_det) / float(side)
        big = cv2.resize(
            aimg,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        big = aimg

    faces2 = app.get(big)
    if faces2:
        best = max(
            faces2,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        emb = get_face_embedding(best)
        if emb is not None:
            return _l2_normalize(emb), False

    emb_fb = get_face_embedding(face)
    if emb_fb is not None:
        print(
            f"[WARN] На aligned-кропе ({label}) recognition и детектор не дали вектор, "
            "косинус по embedding с полного кадра."
        )
        return _l2_normalize(emb_fb), True
    return None, False


@lru_cache(maxsize=1)
def get_lpips_model():
    import lpips

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The parameter 'pretrained' is deprecated.*",
            category=UserWarning,
            module=r"torchvision\.models\._utils",
        )
        warnings.filterwarnings(
            "ignore",
            message="Arguments other than a weight enum or `None` for 'weights' are deprecated.*",
            category=UserWarning,
            module=r"torchvision\.models\._utils",
        )
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
        aligned = warp_face_by_landmarks(image_np, keypoints, output_size)
        if aligned is not None:
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
        bgr_o = cv2.cvtColor(np.array(image_orig_rgb), cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(debug_path / "original_identity_align_bgr.jpg"),
            aligned_face_bgr_for_metrics(bgr_o, face_orig, IDENTITY_ALIGN_SIZE),
        )

    if face_gen is not None:
        get_aligned_face_image(image_gen_rgb, face_gen).save(
            debug_path / "generated_aligned.jpg", quality=95
        )
        crop_face_image(image_gen_rgb, face_gen).save(
            debug_path / "generated_crop.jpg", quality=95
        )
        bgr_g = cv2.cvtColor(np.array(image_gen_rgb), cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(debug_path / "generated_identity_align_bgr.jpg"),
            aligned_face_bgr_for_metrics(bgr_g, face_gen, IDENTITY_ALIGN_SIZE),
        )

    if face_gen_mirrored is not None:
        mirrored_image = mirror_rgb_image(image_gen_rgb)
        get_aligned_face_image(mirrored_image, face_gen_mirrored).save(
            debug_path / "generated_mirrored_aligned.jpg", quality=95
        )
        crop_face_image(mirrored_image, face_gen_mirrored).save(
            debug_path / "generated_mirrored_crop.jpg", quality=95
        )
        bgr_m = cv2.cvtColor(np.array(mirrored_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(debug_path / "generated_mirrored_identity_align_bgr.jpg"),
            aligned_face_bgr_for_metrics(bgr_m, face_gen_mirrored, IDENTITY_ALIGN_SIZE),
        )

    with (debug_path / "metrics_debug.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def compute_composite(
    cos_sim: float | None,
    age_mae: float | None,
    lpips_score: float | None,
) -> float | None:
    """
    Нормализуем каждую метрику в [0, 1] (1 = лучше) и берём взвешенное среднее.
    """
    scores = {}
    weights = {}

    if cos_sim is not None:
        scores["identity"] = (cos_sim + 1) / 2
        weights["identity"] = 0.50

    if age_mae is not None:
        scores["age"] = max(0.0, 1.0 - age_mae / 30.0)
        weights["age"] = 0.30

    if lpips_score is not None:
        scores["lpips"] = max(0.0, 1.0 - lpips_score)
        weights["lpips"] = 0.20

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
    with_lpips: bool = True,
    debug_dir: str | None = None,
    identity_align_size: int = IDENTITY_ALIGN_SIZE,
) -> dict[str, Any]:
    """
    Возвращает словарь метрик. Эту функцию можно вызывать из worker-а.

    Косинусная близость считается по эмбеддингам, извлечённым с **одинаково
    выровненных** кропов (identity_align_size), чтобы original и generated
    были в одном геометрическом виде для ArcFace.
    """
    orig_arg = original_path
    gen_arg = generated_path
    try:
        orig_p = resolve_image_path(original_path, "original")
        gen_p = resolve_image_path(generated_path, "generated")
    except FileNotFoundError as e:
        raise ValueError(str(e)) from e

    raw_o = Path(orig_arg).expanduser()
    raw_g = Path(gen_arg).expanduser()
    if not raw_o.is_file():
        print(f"[INFO] original: {orig_arg!r} → {orig_p}")
    if not raw_g.is_file():
        print(f"[INFO] generated: {gen_arg!r} → {gen_p}")

    try:
        image_orig_rgb = load_rgb_image(str(orig_p))
        image_gen_rgb = load_rgb_image(str(gen_p))
    except Exception as e:
        raise ValueError(f"Не удалось загрузить изображения: {e}") from e

    img_orig = cv2.cvtColor(np.array(image_orig_rgb), cv2.COLOR_RGB2BGR)
    img_gen = cv2.cvtColor(np.array(image_gen_rgb), cv2.COLOR_RGB2BGR)

    cos_sim = None
    predicted_age = None
    age_mae = None
    face_orig = None
    face_gen = None
    face_gen_mirrored = None
    mirrored_cos_sim = None
    identity_cos_used_fallback = False
    det_used_original: int | None = None
    det_used_generated: int | None = None

    try:
        app = get_face_app(det_size)
        face_orig, det_used_original = get_main_face_multiscale(
            img_orig, "original", det_size
        )
        face_gen, det_used_generated = get_main_face_multiscale(
            img_gen, "generated", det_size
        )

        if face_orig is not None and face_gen is not None:
            emb_orig, fb_o = compute_metric_embedding(
                app, img_orig, face_orig, "original", align_size=identity_align_size
            )
            emb_gen, fb_g = compute_metric_embedding(
                app, img_gen, face_gen, "generated", align_size=identity_align_size
            )
            identity_cos_used_fallback = fb_o or fb_g
            if emb_orig is not None and emb_gen is not None:
                cos_sim = cosine_similarity(emb_orig, emb_gen)

            predicted_age = get_face_age(face_gen)
            if predicted_age is not None:
                age_mae = abs(predicted_age - target_age)

        mirrored_gen_rgb = mirror_rgb_image(image_gen_rgb)
        mirrored_gen_bgr = cv2.cvtColor(np.array(mirrored_gen_rgb), cv2.COLOR_RGB2BGR)
        face_gen_mirrored, _ = get_main_face_multiscale(
            mirrored_gen_bgr, "generated_mirrored", det_size
        )
        if face_orig is not None and face_gen_mirrored is not None:
            emb_orig, fb_o = compute_metric_embedding(
                app, img_orig, face_orig, "original", align_size=identity_align_size
            )
            emb_gen_m, fb_m = compute_metric_embedding(
                app,
                mirrored_gen_bgr,
                face_gen_mirrored,
                "generated_mirrored",
                align_size=identity_align_size,
            )
            if fb_o or fb_m:
                identity_cos_used_fallback = True
            if emb_orig is not None and emb_gen_m is not None:
                mirrored_cos_sim = cosine_similarity(emb_orig, emb_gen_m)
    except ImportError:
        print("[WARN] insightface не установлен. Пропускаем identity/age metrics.")
    except Exception as e:
        print(f"[WARN] InsightFace metrics failed: {e}")

    lpips_score = None
    if with_lpips:
        if face_orig is not None and face_gen is not None:
            aligned_orig = get_aligned_face_image(image_orig_rgb, face_orig)
            aligned_gen = get_aligned_face_image(image_gen_rgb, face_gen)
            lpips_score = compute_lpips_images(aligned_orig, aligned_gen)
        else:
            print("[WARN] Face alignment unavailable. LPIPS uses full resized images.")
            lpips_score = compute_lpips(str(orig_p), str(gen_p))

    composite = compute_composite(cos_sim, age_mae, lpips_score)

    metrics = {
        "cosine_similarity": cos_sim,
        "cosine_similarity_mirrored_generated": mirrored_cos_sim,
        "predicted_age": predicted_age,
        "target_age": target_age,
        "age_mae": age_mae,
        "lpips": lpips_score,
        "composite_score": composite,
        "metrics_json": {
            "original_path": str(orig_p),
            "generated_path": str(gen_p),
            "original_path_as_given": orig_arg,
            "generated_path_as_given": gen_arg,
            "det_size": det_size,
            "det_used_original": det_used_original,
            "det_used_generated": det_used_generated,
            "identity_align_size": identity_align_size,
            "identity_cosine_used_fullframe_fallback": identity_cos_used_fallback,
            "with_lpips": with_lpips,
            "original_face": face_to_debug_dict(face_orig),
            "generated_face": face_to_debug_dict(face_gen),
            "generated_mirrored_face": face_to_debug_dict(face_gen_mirrored),
            "alignment_note": ALIGNMENT_NOTE,
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
    mj = metrics.get("metrics_json") or {}
    note = mj.get("alignment_note", ALIGNMENT_NOTE)
    print("  Примечание (геометрия):")
    for chunk_start in range(0, len(note), 72):
        print(f"    {note[chunk_start : chunk_start + 72]}")

    align_sz = mj.get("identity_align_size", IDENTITY_ALIGN_SIZE)

    cos_sim = metrics["cosine_similarity"]
    if cos_sim is not None:

        print(
            f"  Cosine Similarity (идентичность) : {cos_sim:.4f} "
            f"[aligned {align_sz}×{align_sz}]"
        )
    else:
        print("  Cosine Similarity                : N/A")
        of = mj.get("original_face")
        gf = mj.get("generated_face")
        if gf is None:
            print("       → на generated InsightFace не нашёл лицо (файл при этом открыт)")
        if of is None:
            print("       → на original InsightFace не нашёл лицо")


    predicted_age = metrics["predicted_age"]
    age_mae = metrics["age_mae"]
    if predicted_age is not None and age_mae is not None:
        print(f"  Predicted age                    : {predicted_age:.1f} лет")
        print(f"  Target age                       : {metrics['target_age']} лет")
        print(f"  Age MAE                          : {age_mae:.1f} лет ")
    else:
        print("  Age MAE                          : N/A")

    lpips_score = metrics["lpips"]
    if lpips_score is not None:

        print(f"  LPIPS (lower is better)          : {lpips_score:.4f} ")
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
            with_lpips=not args.skip_lpips,
            debug_dir=args.debug_dir,
        )
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")

    print_results(metrics)


if __name__ == "__main__":
    main()
