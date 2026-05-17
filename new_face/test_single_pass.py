import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
from diffusers import AutoPipelineForImage2Image
from new_face.age_run_config import (
    load_age_prompt_pair_or_fallback,
    merge_run_config,
    resolve_age_coefficients,
)
from new_face.evaluate_face import (
    estimate_insightface_source_age,
    evaluate_face,
    print_results,
)
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

_REPO = Path(__file__).resolve().parent
_MODELS_DIR = Path(os.getenv("YNF_MODELS_DIR", _REPO / "models"))


CONFIG = {
    "clip": os.getenv(
        "YNF_CLIP_PATH", str(_MODELS_DIR / "CLIP-ViT-H-14-laion2B-s32B-b79K")
    ),
    "sdxl_model": os.getenv("YNF_SDXL_MODEL_PATH", str(_MODELS_DIR / "RealVisXL_V5.0")),
    "ip_adapter_repo": os.getenv("YNF_IP_ADAPTER_REPO", str(_MODELS_DIR / "Adapter")),
    "ip_adapter_weight": os.getenv(
        "YNF_IP_ADAPTER_WEIGHT", "ip-adapter-plus-face_sdxl_vit-h.safetensors"
    ),
    "lora_mode": os.getenv("YNF_LORA_MODE", "custom"),  # none | custom | age_slider
    "custom_lora": os.getenv(
        "YNF_CUSTOM_LORA_PATH", str(_MODELS_DIR / "Lora" / "HQ_mix_big_RealVZ.safetensors")
    ),
    "age_slider_lora": os.getenv(
        "YNF_AGE_SLIDER_LORA_PATH", str(_MODELS_DIR / "Lora" / "age_slider-sdxl.safetensors")
    ),
    "custom_lora_scale": 0.5,
    "age_slider_scale": 10.0,  # negative = younger, positive = older
    "input_image": None,
    "output_image": None,
    "output_dir": os.getenv("YNF_OUTPUT_DIR", "/tmp/your-new-face/outputs"),
    "metrics_file": os.getenv(
        "YNF_METRICS_FILE", "/tmp/your-new-face/single_pass_metrics.jsonl"
    ),
    "device": os.getenv("YNF_DEVICE", "cuda"),
    "dtype": torch.float16,
    "size": 1024,
    # Если список не пустой — генерируем по каждому возрасту (один pipeline, одна копия source).
    "target_ages": [],
    # "target_ages": [20, 30, 40, 50, 60, 70],
    "target_age": 40,  # используется только когда target_ages пустой или None
    "gender": os.getenv("YNF_DEFAULT_GENDER", "male"),
    "seed": int(os.getenv("YNF_SEED", "8888")),
    "steps": 15,
    "cfg": 7.0,
    "strength": 0.3,
    "ip_adapter_scale": 0.9,
    # Если None — negative из JSON/фолбэка; иначе фиксированная строка для всех возрастов.
    "negative_prompt": None,
    "use_age_prompt_files": True,
    "age_prompts_dir": str(_REPO / "data" / "age_prompts"),
    "age_coefficients_path": str(_REPO / "data" / "age_coefficients.json"),
    # Исходный возраст для age_coefficients.json: None → InsightFace по input_image;
    # число — зафиксировать. infer_source_age=False и None — не оценивать (как раньше).
    "source_age": None,
    "infer_source_age": os.getenv("YNF_INFER_SOURCE_AGE", "true").lower() == "true",
    "insightface_age_det_size": 640,
}

_PIPELINE = None
_PIPELINE_KEY = None


def build_negative_prompt(age: int) -> str:
    common = (
        "lowres, blurry, deformed, disfigured, bad anatomy, watermark, "
        "jpeg artifacts, plastic skin, cartoon, 3d render"
    )
    if age < 45:
        extra = "old, elderly, wrinkles, aged face, gray hair, sagging skin, "
    else:
        extra = "young, youthful face, smooth skin, baby face, teenager, "
    return extra + common


def resolve_target_ages():
    ages = CONFIG.get("target_ages")
    if ages:
        return [int(a) for a in ages]
    return [int(CONFIG["target_age"])]


def load_img(path):
    return Image.open(path).convert("RGB").resize((CONFIG["size"], CONFIG["size"]))


def get_source_image_info(path):
    image = Image.open(path)
    suffix = Path(path).suffix.lower()
    return image.size, image.format, suffix


def get_output_path(input_path, target_age: int, run=None):
    if CONFIG["output_image"]:
        return CONFIG["output_image"]

    input_file = Path(input_path)
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / f"{input_file.stem}_{get_run_tag(target_age, run)}{input_file.suffix}")


def get_source_copy_path(input_path):
    """Одна копия исходника на сессию (имя без возраста)."""
    input_file = Path(input_path)
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / f"{input_file.stem}_source{input_file.suffix}")


def format_float(value):
    return str(value).replace(".", "p").replace("-", "m")


def get_run_tag(target_age: int, run=None):
    cfg = run if run is not None else CONFIG
    _, lora_scale, adapter_name = get_lora_config(cfg)
    lora_part = "lora_none"
    if adapter_name:
        lora_part = f"lora_{adapter_name}_{format_float(lora_scale)}"

    return (
        f"single_age{target_age}"
        f"_str{format_float(cfg['strength'])}"
        f"_cfg{format_float(cfg['cfg'])}"
        f"_ip{format_float(cfg['ip_adapter_scale'])}"
        f"_{lora_part}"
        f"_steps{cfg['steps']}"
        f"_seed{cfg['seed']}"
    )


def save_like_source(image, input_path, output_path):
    source_size, source_format, suffix = get_source_image_info(input_path)
    resized = image.convert("RGB").resize(source_size, Image.LANCZOS)

    save_kwargs = {}
    if source_format == "JPEG" or suffix in (".jpg", ".jpeg"):
        save_kwargs.update({"quality": 95, "subsampling": 0})
    resized.save(output_path, format=source_format, **save_kwargs)


def copy_source_once(input_path):
    source_copy = get_source_copy_path(input_path)
    if not os.path.exists(source_copy):
        shutil.copy2(input_path, source_copy)
    return source_copy


def get_age_group(age):
    if age < 25:
        return "young adult"
    if age < 30:
        return "young adult"
    if age < 40:
        return "adult"
    if age < 50:
        return "middle-aged"
    if age < 60:
        return "older adult"
    return "elderly"


def build_prompt(age, gender):
    if age >= 70:
        return (
            f"portrait photo of {get_age_group(age)} {gender}, age {int(age)}, "
            "realistic elderly face, very old face, deep wrinkles, forehead wrinkles, "
            "crow's feet, nasolabial folds, under eye bags, aged skin texture, "
            "sagging skin, gray hair"
        )

    if age >= 50:
        return (
            f"portrait photo of {get_age_group(age)} {gender}, age {int(age)}, "
            "realistic aging, mature face, deep wrinkles, forehead wrinkles, "
            "crow's feet, nasolabial folds, under eye bags, aged skin texture, "
            "gray hair"
        )

    return (
        f"portrait photo of {get_age_group(age)} {gender}, age {int(age)}, "
        "youthful face, smooth skin"
    )


def get_lora_config(run=None):
    cfg = run if run is not None else CONFIG
    mode = cfg["lora_mode"]
    if mode == "none":
        return None, None, None
    if mode == "custom":
        return CONFIG["custom_lora"], float(cfg["custom_lora_scale"]), "custom"
    if mode == "age_slider":
        return CONFIG["age_slider_lora"], float(cfg["age_slider_scale"]), "age_slider"
    raise ValueError(f"Unknown lora_mode: {mode}")


def append_metrics(
    metrics,
    output_image,
    source_copy,
    prompt,
    target_age: int,
    negative_prompt: str,
    run=None,
    *,
    source_age_used=None,
    source_age_meta=None,
):
    cfg = run if run is not None else CONFIG
    metrics_path = Path(CONFIG["metrics_file"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    lora_path, lora_scale, adapter_name = get_lora_config(cfg)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_tag": get_run_tag(target_age, cfg),
        "input_image": CONFIG["input_image"],
        "source_age_config": CONFIG.get("source_age"),
        "source_age_used": source_age_used,
        "source_age_meta": source_age_meta,
        "source_copy": source_copy,
        "output_image": output_image,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "target_age": target_age,
        "gender": CONFIG["gender"],
        "seed": cfg["seed"],
        "steps": cfg["steps"],
        "cfg": cfg["cfg"],
        "strength": cfg["strength"],
        "ip_adapter_weight": CONFIG["ip_adapter_weight"],
        "ip_adapter_scale": cfg["ip_adapter_scale"],
        "lora_mode": CONFIG["lora_mode"],
        "lora_path": lora_path,
        "lora_adapter_name": adapter_name,
        "lora_scale": lora_scale,
        "cosine_similarity": metrics["cosine_similarity"],
        "cosine_similarity_mirrored_generated": metrics[
            "cosine_similarity_mirrored_generated"
        ],
        "predicted_age": metrics["predicted_age"],
        "age_mae": metrics["age_mae"],
        "lpips": metrics["lpips"],
        "composite_score": metrics["composite_score"],
    }

    with metrics_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(metrics_path)


def load_pipeline():
    _validate_model_paths()
    pipe = AutoPipelineForImage2Image.from_pretrained(
        CONFIG["sdxl_model"],
        torch_dtype=CONFIG["dtype"],
        variant="fp16",
    )
    pipe.to(CONFIG["device"])

    pipe.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        CONFIG["clip"],
        torch_dtype=CONFIG["dtype"],
    ).to(CONFIG["device"])
    pipe.feature_extractor = CLIPImageProcessor.from_pretrained(CONFIG["clip"])

    print("Loading IP-Adapter:", CONFIG["ip_adapter_weight"])
    pipe.load_ip_adapter(
        CONFIG["ip_adapter_repo"],
        subfolder="sdxl_models",
        weight_name=CONFIG["ip_adapter_weight"],
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(CONFIG["ip_adapter_scale"])

    lora_path, lora_scale, adapter_name = get_lora_config()
    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA ({adapter_name}):", lora_path)
        print("LoRA scale:", lora_scale)
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
    else:
        print("LoRA: disabled or file not found")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        print(f"[WARN] xformers attention disabled: {exc}")
    pipe.vae.enable_slicing()
    pipe.vae.to(CONFIG["dtype"])
    return pipe


def get_pipeline():
    global _PIPELINE, _PIPELINE_KEY

    pipeline_key = (
        CONFIG["clip"],
        CONFIG["sdxl_model"],
        CONFIG["ip_adapter_repo"],
        CONFIG["ip_adapter_weight"],
        CONFIG["lora_mode"],
        CONFIG["custom_lora"],
        CONFIG["age_slider_lora"],
        CONFIG["device"],
    )
    if _PIPELINE is None or _PIPELINE_KEY != pipeline_key:
        print("[INFO] Loading generation pipeline into memory...")
        _PIPELINE = load_pipeline()
        _PIPELINE_KEY = pipeline_key
        print("[INFO] Generation pipeline is ready and cached.")

    return _PIPELINE


def _validate_model_paths() -> None:
    required = {
        "CLIP": CONFIG["clip"],
        "SDXL model": CONFIG["sdxl_model"],
        "IP-Adapter repo": CONFIG["ip_adapter_repo"],
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Не найдены ML-модели внутри контейнера. Проверь YNF_* env или скопируй модели "
            "в new_face/models перед сборкой:\n  " + "\n  ".join(missing)
        )


def apply_inference_overrides(pipe, run) -> None:
    """Подставляет strength/cfg/ip/steps/lora scale из merged run (после age_coefficients)."""
    pipe.set_ip_adapter_scale(float(run["ip_adapter_scale"]))
    _, scale, adapter_name = get_lora_config(run)
    if adapter_name:
        pipe.set_adapters([adapter_name], adapter_weights=[float(scale)])


def run_single_pass(
    *,
    input_path: str,
    output_path: str,
    target_age: int,
    gender: str | None = None,
    task_id: str | None = None,
):
    run_config = dict(CONFIG)
    run_config["input_image"] = input_path
    run_config["output_image"] = output_path
    run_config["target_age"] = int(target_age)
    run_config["target_ages"] = []
    if gender:
        run_config["gender"] = gender
    if task_id:
        run_config["metrics_file"] = str(Path(run_config["output_dir"]) / f"{task_id}_metrics.jsonl")

    return _run_configured_single_pass(run_config)


def _run_configured_single_pass(run_config: dict):
    previous_config = dict(CONFIG)
    CONFIG.update(run_config)
    try:
        return _main_impl()
    finally:
        CONFIG.clear()
        CONFIG.update(previous_config)


def _main_impl():
    ages = resolve_target_ages()
    input_path = CONFIG["input_image"]
    if not input_path:
        raise ValueError("CONFIG['input_image'] must be set")
    init_image = load_img(input_path)
    pipe = get_pipeline()
    source_copy = copy_source_once(input_path)

    coef_path = CONFIG.get("age_coefficients_path") or None
    prompts_dir = CONFIG["age_prompts_dir"] if CONFIG.get("use_age_prompt_files") else None

    source_age_used = CONFIG.get("source_age")
    source_age_meta = None
    if source_age_used is None and CONFIG.get("infer_source_age", True):
        det_sz = int(CONFIG.get("insightface_age_det_size", 640))
        source_age_used, source_age_meta = estimate_insightface_source_age(
            input_path, det_size=det_sz
        )
        if source_age_used is not None:
            print(
                f"[INFO] source_age (InsightFace) ≈ {source_age_used:.1f}, "
                f"det={source_age_meta.get('det_used')}"
            )
        else:
            print(
                "[WARN] source_age: на входе лицо не найдено — "
                "коэффициенты перехода считаются без исходного возраста"
            )
    elif isinstance(source_age_used, (int, float)):
        source_age_used = float(source_age_used)
        print(f"[INFO] source_age (из CONFIG): {source_age_used}")
    else:
        source_age_used = None

    print("Target ages:", ages)
    print("Strength:", CONFIG["strength"], "| IP-Adapter:", CONFIG["ip_adapter_scale"])
    print("Age prompts dir:", prompts_dir or "(встроенные)")
    print("Age coefficients:", coef_path or "(выкл)")
    print("source_age для коэффициентов:", source_age_used)
    print("Source copy:", source_copy)

    results = []
    for target_age in ages:
        coef = resolve_age_coefficients(
            coef_path,
            source_age=source_age_used,
            target_age=target_age,
            gender=CONFIG["gender"],
        )
        run = merge_run_config(CONFIG, coef)
        apply_inference_overrides(pipe, run)

        prompt, neg_from_file = load_age_prompt_pair_or_fallback(
            prompts_dir,
            target_age,
            CONFIG["gender"],
            build_prompt,
            build_negative_prompt,
        )
        negative = (
            CONFIG["negative_prompt"]
            if CONFIG.get("negative_prompt")
            else neg_from_file
        )

        print("\n" + "=" * 55)
        print(f"target_age={target_age}")
        print("Run overrides:", {k: run[k] for k in ("strength", "cfg", "ip_adapter_scale", "steps") if k in run})
        print("Prompt:", prompt)
        print("Negative:", negative[:120] + ("…" if len(negative) > 120 else ""))

        generator = torch.Generator(device=CONFIG["device"]).manual_seed(int(run["seed"]))
        image = pipe(
            prompt,
            negative_prompt=negative,
            image=init_image,
            ip_adapter_image=init_image,
            num_inference_steps=int(run["steps"]),
            strength=float(run["strength"]),
            guidance_scale=float(run["cfg"]),
            generator=generator,
        ).images[0]

        output_image = get_output_path(input_path, target_age, run)
        if len(ages) > 1 and CONFIG.get("output_image"):
            raise ValueError(
                "При нескольких target_ages укажи output_image=None, иначе все возрасты "
                "перезапишут один файл."
            )
        save_like_source(image, Path(input_path), Path(output_image))

        metrics = evaluate_face(
            original_path=input_path,
            generated_path=output_image,
            target_age=target_age,
            with_lpips=True,
        )
        metrics_file = append_metrics(
            metrics,
            output_image,
            source_copy,
            prompt,
            target_age,
            negative,
            run,
            source_age_used=source_age_used,
            source_age_meta=source_age_meta,
        )
        print_results(metrics)
        print("Saved:", output_image)
        print("Metrics log:", metrics_file)
        results.append(
            {
                "output_image": output_image,
                "source_copy": source_copy,
                "metrics": metrics,
                "metrics_file": metrics_file,
                "prompt": prompt,
                "negative_prompt": negative,
                "run": run,
                "source_age_used": source_age_used,
                "source_age_meta": source_age_meta,
            }
        )

        del image
        torch.cuda.empty_cache()

    return results[0] if len(results) == 1 else results


def main():
    return _main_impl()


if __name__ == "__main__":
    main()
