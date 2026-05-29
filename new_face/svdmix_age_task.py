from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from new_face.evaluate_face import estimate_insightface_source_age, evaluate_face

_REPO = Path(__file__).resolve().parent
_MODELS_DIR = Path(os.getenv("YNF_MODELS_DIR", _REPO / "models"))
_DATA_DIR = _REPO / "data" / "svdmix"

LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
GROUP_ORDER = ["20_24", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60_"]
PAIR_SUFFIXES = (
    (".lora.down.weight", ".lora.up.weight"),
    (".lora_down.weight", ".lora_up.weight"),
    (".lora_A.weight", ".lora_B.weight"),
)

CONFIG: dict[str, Any] = {
    "base_model": os.getenv("YNF_SDXL_MODEL_PATH", str(_MODELS_DIR / "RealVisXL_V5.0")),
    "clip": os.getenv(
        "YNF_CLIP_PATH", str(_MODELS_DIR / "CLIP-ViT-H-14-laion2B-s32B-b79K")
    ),
    "ip_adapter_repo": os.getenv("YNF_IP_ADAPTER_REPO", str(_MODELS_DIR / "Adapter")),
    "ip_adapter_weight": os.getenv(
        "YNF_IP_ADAPTER_WEIGHT", "ip-adapter-plus-face_sdxl_vit-h.safetensors"
    ),
    "young_lora": os.getenv(
        "YNF_SVDMIX_YOUNG_LORA", str(_MODELS_DIR / "Lora" / "AB_young.safetensors")
    ),
    "old_lora": os.getenv(
        "YNF_SVDMIX_OLD_LORA", str(_MODELS_DIR / "Lora" / "AB_old.safetensors")
    ),
    "refined_params_json": os.getenv(
        "YNF_SVDMIX_REFINED_PARAMS", str(_DATA_DIR / "refined_selected_params.json")
    ),
    "coefficients_json": os.getenv(
        "YNF_SVDMIX_COEFFICIENTS", str(_DATA_DIR / "age_transition_coefficients_by_group.json")
    ),
    "prompts_json": os.getenv(
        "YNF_SVDMIX_PROMPTS", str(_DATA_DIR / "age_group_prompts.json")
    ),
    "fused_lora_cache": os.getenv(
        "YNF_SVDMIX_FUSED_CACHE", "/tmp/your-new-face/svdmix_fused_loras"
    ),
    "precomputed_lora_dir": os.getenv(
        "YNF_SVDMIX_PRECOMPUTED_DIR",
        str(_MODELS_DIR / "Lora" / "svdmix_fused_age_grid"),
    ),
    "use_precomputed_lora": os.getenv("YNF_SVDMIX_USE_PRECOMPUTED", "true").lower()
    == "true",
    "age_grid_min": float(os.getenv("YNF_SVDMIX_AGE_GRID_MIN", "20")),
    "age_grid_max": float(os.getenv("YNF_SVDMIX_AGE_GRID_MAX", "60")),
    "age_grid_step": float(os.getenv("YNF_SVDMIX_AGE_GRID_STEP", "2.5")),
    "device": os.getenv("YNF_DEVICE", "cuda"),
    "text_encoder_device": os.getenv("YNF_TEXT_ENCODER_DEVICE", "cpu"),
    "dtype": torch.float16,
    "image_size": int(os.getenv("YNF_IMAGE_SIZE", "1024")),
    "steps": int(os.getenv("YNF_SVDMIX_STEPS", "20")),
    "seed": int(os.getenv("YNF_SEED", "8888")),
    "gender": os.getenv("YNF_DEFAULT_GENDER", "male"),
    "default_source_group": os.getenv("YNF_DEFAULT_SOURCE_GROUP", "30_34"),
    "infer_source_age": os.getenv("YNF_INFER_SOURCE_AGE", "true").lower() == "true",
}

_PIPELINE = None
_PIPELINE_KEY = None
_LOADED_ADAPTERS: set[str] = set()
_YOUNG_STATE = None
_OLD_STATE = None


def run_svdmix_age_transform(
    *,
    image_path: str,
    output_path: str | None = None,
    gender: str | None = None,
    source_group: str | None = None,
    target_group: str | None = None,
    source_age: int | float | None = None,
    target_age: int | float | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
    with_metrics: bool = True,
) -> dict[str, Any]:
    gender = (gender or CONFIG["gender"]).lower().strip()
    if gender not in {"male", "female"}:
        raise ValueError(f"gender must be male/female, got: {gender!r}")

    if source_group is None:
        source_age = source_age if source_age is not None else _estimate_source_age(image_path)
        source_group = group_from_age(source_age) if source_age is not None else CONFIG["default_source_group"]
    else:
        source_group = normalize_group(source_group)

    if target_group is None:
        if target_age is None:
            raise ValueError("target_group or target_age must be provided")
        target_group = group_from_age(target_age)
    else:
        target_group = normalize_group(target_group)

    output_path = str(_resolve_output_path(image_path, output_path, output_dir))
    seed = int(seed if seed is not None else CONFIG["seed"])

    preset, parameter_source = select_parameters(gender, source_group, target_group)
    coeff = preset["coefficients"]
    prompt, negative_prompt = select_prompts(preset, target_group, gender)
    requested_lora_age = float(target_age or group_anchor_age(target_group))
    fused_lora_dir, lora_age, alpha_young = resolve_fused_lora_for_age(
        requested_lora_age,
        fallback_alpha=float(preset["alpha_young"]),
    )
    lora_scale = float(preset["lora_scale"])

    pipe = get_pipeline()
    adapter_name = load_fused_lora(
        pipe,
        fused_lora_dir,
        alpha_young,
        lora_scale,
        float(coeff["ip_adapter_scale"]),
    )
    try:
        prompt_kwargs = encode_prompt_on_cpu(pipe, prompt, negative_prompt)
        init_image = load_image(image_path)
        generator = torch.Generator(device=CONFIG["device"]).manual_seed(seed)
        image = pipe(
            prompt=None,
            negative_prompt=None,
            image=init_image,
            ip_adapter_image=init_image,
            num_inference_steps=int(coeff.get("steps") or CONFIG["steps"]),
            strength=float(coeff["strength"]),
            guidance_scale=float(coeff["cfg"]),
            generator=generator,
            **prompt_kwargs,
        ).images[0]

        save_like_source(image, image_path, output_path)
        del image
    finally:
        unload_fused_lora(pipe, adapter_name)
        torch.cuda.empty_cache()

    metrics = None
    if with_metrics:
        metrics = evaluate_face(
            original_path=image_path,
            generated_path=output_path,
            target_age=int(target_age or group_anchor_age(target_group)),
            with_lpips=True,
        )

    metadata = {
        "source_image": str(image_path),
        "output_image": str(output_path),
        "gender": gender,
        "source_group": source_group,
        "target_group": target_group,
        "source_age": source_age,
        "target_age": target_age,
        "requested_lora_age": requested_lora_age,
        "selected_lora_age": lora_age,
        "alpha_young": alpha_young,
        "lora_scale": lora_scale,
        "strength": float(coeff["strength"]),
        "ip_adapter_scale": float(coeff["ip_adapter_scale"]),
        "cfg": float(coeff["cfg"]),
        "steps": int(coeff.get("steps") or CONFIG["steps"]),
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "selected_parameter_source": parameter_source,
        "lora_source": "precomputed_age_grid"
        if CONFIG["use_precomputed_lora"]
        else "runtime_svdmix",
        "fused_lora_dir": str(fused_lora_dir),
        "metrics": metrics,
    }
    metadata_path = Path(output_path).with_suffix(Path(output_path).suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def get_pipeline():
    global _PIPELINE, _PIPELINE_KEY
    key = (
        CONFIG["base_model"],
        CONFIG["clip"],
        CONFIG["ip_adapter_repo"],
        CONFIG["ip_adapter_weight"],
        CONFIG["device"],
        CONFIG["text_encoder_device"],
    )
    if _PIPELINE is None or _PIPELINE_KEY != key:
        validate_paths(
            {
                "SDXL model": CONFIG["base_model"],
                "CLIP vision": CONFIG["clip"],
                "IP-Adapter repo": CONFIG["ip_adapter_repo"],
            }
        )
        pipe = AutoPipelineForImage2Image.from_pretrained(
            CONFIG["base_model"],
            torch_dtype=CONFIG["dtype"],
            variant="fp16",
        ).to(CONFIG["device"])
        pipe.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            CONFIG["clip"], torch_dtype=CONFIG["dtype"]
        ).to(CONFIG["device"])
        pipe.feature_extractor = CLIPImageProcessor.from_pretrained(CONFIG["clip"])
        pipe.load_ip_adapter(
            CONFIG["ip_adapter_repo"],
            subfolder="sdxl_models",
            weight_name=CONFIG["ip_adapter_weight"],
            image_encoder_folder=None,
        )
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            print(f"[WARN] xformers attention disabled: {exc}")
        pipe.vae.enable_slicing()
        pipe.vae.to(CONFIG["dtype"])
        move_text_encoders(pipe, CONFIG["text_encoder_device"])
        _PIPELINE = pipe
        _PIPELINE_KEY = key
        _LOADED_ADAPTERS.clear()
    return _PIPELINE


def select_parameters(gender: str, source_group: str, target_group: str) -> tuple[dict[str, Any], str]:
    key = f"{gender}|{source_group}|{target_group}"
    refined = load_json(CONFIG["refined_params_json"])
    if key in refined.get("transitions", {}):
        row = refined["transitions"][key]
        return {
            **row,
            "coefficients": {
                **row["coefficients"],
                "steps": int(row["coefficients"].get("steps") or CONFIG["steps"]),
            },
        }, "refined_selected_params"

    coeffs = load_json(CONFIG["coefficients_json"])
    transition = coeffs["coefficients"][gender][source_group]["targets"][target_group]
    rec = transition["recommended"]
    alpha_young = alpha_for_target(target_group)
    lora_scale = float(rec["custom_lora_scale"])
    return {
        "gender": gender,
        "source_group": source_group,
        "target_group": target_group,
        "alpha_young": alpha_young,
        "lora_scale": lora_scale,
        "coefficients": {
            "strength": float(rec["strength"]),
            "ip_adapter_scale": float(rec["ip_adapter_scale"]),
            "cfg": float(rec["cfg"]),
            "alpha_young": alpha_young,
            "lora_scale": lora_scale,
            "steps": int(rec.get("steps") or CONFIG["steps"]),
        },
        "prompt": None,
        "negative_prompt": None,
    }, "base_coefficients"


def select_prompts(preset: dict[str, Any], target_group: str, gender: str) -> tuple[str, str]:
    prompt = preset.get("prompt")
    negative = preset.get("negative_prompt")
    if prompt and negative:
        return prompt, negative

    prompts = load_json(CONFIG["prompts_json"])
    data = prompts[target_group]
    return data["positive"].replace("{gender}", gender), data["negative"].replace("{gender}", gender)


def ensure_fused_lora(alpha: float) -> Path:
    cache_dir = Path(CONFIG["fused_lora_cache"])
    out_dir = cache_dir / alpha_cache_key(alpha)
    out_file = out_dir / LORA_WEIGHT_NAME
    if out_file.is_file():
        return out_dir

    young_path = resolve_lora_path(CONFIG["young_lora"])
    old_path = resolve_lora_path(CONFIG["old_lora"])
    young_state, old_state = get_lora_states()
    fused_state = fuse_lora_state_dict(young_state, old_state, float(alpha))
    save_fused_lora(
        fused_state,
        out_dir,
        {
            "agebooth.style": "svdmix_per_factor",
            "agebooth.alpha_young": f"{float(alpha):.6f}",
            "agebooth.young_lora": str(young_path),
            "agebooth.old_lora": str(old_path),
        },
    )
    return out_dir


def resolve_fused_lora_for_age(
    target_age: float,
    *,
    fallback_alpha: float,
) -> tuple[Path, float, float]:
    if not CONFIG["use_precomputed_lora"]:
        return ensure_fused_lora(fallback_alpha), target_age, fallback_alpha

    lora_age = nearest_grid_age(target_age)
    alpha = alpha_for_age(lora_age)
    lora_dir = precomputed_lora_dir_for_age(lora_age)
    lora_file = lora_dir / LORA_WEIGHT_NAME
    if not lora_file.is_file():
        raise FileNotFoundError(
            "Precomputed SVDMix LoRA is missing. Run precompute first: "
            f"{lora_file}"
        )
    return lora_dir, lora_age, alpha


def precomputed_lora_dir_for_age(age: float) -> Path:
    return Path(CONFIG["precomputed_lora_dir"]) / age_cache_key(age)


def age_grid_values() -> list[float]:
    current = float(CONFIG["age_grid_min"])
    max_age = float(CONFIG["age_grid_max"])
    step = float(CONFIG["age_grid_step"])
    values = []
    while current <= max_age + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def nearest_grid_age(age: float) -> float:
    return min(age_grid_values(), key=lambda item: abs(item - float(age)))


def alpha_for_age(age: float) -> float:
    lo = float(CONFIG["age_grid_min"])
    hi = float(CONFIG["age_grid_max"])
    alpha = 1.0 - ((float(age) - lo) / (hi - lo))
    return round(max(0.0, min(1.0, alpha)), 4)


def age_cache_key(age: float) -> str:
    return f"age_{format_float(round(float(age), 4))}"


def load_fused_lora(
    pipe,
    fused_lora_dir: Path,
    alpha: float,
    lora_scale: float,
    ip_adapter_scale: float,
) -> str:
    pipe.set_ip_adapter_scale(float(ip_adapter_scale))
    adapter_name = alpha_cache_key(alpha)
    if adapter_name in _LOADED_ADAPTERS:
        unload_fused_lora(pipe, adapter_name)
    if adapter_name not in _LOADED_ADAPTERS:
        pipe.load_lora_weights(str(fused_lora_dir), adapter_name=adapter_name)
        _LOADED_ADAPTERS.add(adapter_name)
    pipe.set_adapters([adapter_name], adapter_weights=[float(lora_scale)])
    move_text_encoders(pipe, CONFIG["text_encoder_device"])
    return adapter_name


def unload_fused_lora(pipe, adapter_name: str) -> None:
    if adapter_name not in _LOADED_ADAPTERS:
        return
    try:
        if hasattr(pipe, "delete_adapters"):
            pipe.delete_adapters(adapter_name)
        elif hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
        else:
            return
    finally:
        _LOADED_ADAPTERS.discard(adapter_name)
        move_text_encoders(pipe, CONFIG["text_encoder_device"])


def move_text_encoders(pipe, device: str) -> None:
    for name in ("text_encoder", "text_encoder_2"):
        encoder = getattr(pipe, name, None)
        if encoder is not None:
            encoder.to(device)


def encode_prompt_on_cpu(pipe, prompt: str, negative_prompt: str) -> dict[str, torch.Tensor]:
    text_device = torch.device(CONFIG["text_encoder_device"])
    target_device = torch.device(CONFIG["device"])
    move_text_encoders(pipe, str(text_device))
    with torch.inference_mode():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=text_device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
        )
    return {
        "prompt_embeds": prompt_embeds.to(device=target_device, dtype=CONFIG["dtype"]),
        "negative_prompt_embeds": negative_prompt_embeds.to(
            device=target_device, dtype=CONFIG["dtype"]
        ),
        "pooled_prompt_embeds": pooled_prompt_embeds.to(
            device=target_device, dtype=CONFIG["dtype"]
        ),
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds.to(
            device=target_device, dtype=CONFIG["dtype"]
        ),
    }

def fuse_lora_state_dict(
    young: dict[str, torch.Tensor],
    old: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    young_pairs = find_lora_pairs(young)
    old_pairs = find_lora_pairs(old)
    prefixes = common_pair_prefixes(young_pairs, old_pairs)
    fused = {key: value.clone() for key, value in young.items()}
    for prefix in prefixes:
        y_down_key, y_up_key, _, _ = young_pairs[prefix]
        o_down_key, o_up_key, _, _ = old_pairs[prefix]
        fused[y_down_key] = svdmix(young[y_down_key], old[o_down_key], alpha).contiguous()
        fused[y_up_key] = svdmix(young[y_up_key], old[o_up_key], alpha).contiguous()
    return fused


def find_lora_pairs(state_dict: dict[str, torch.Tensor]) -> dict[str, tuple[str, str, str, str]]:
    pairs = {}
    for down_suffix, up_suffix in PAIR_SUFFIXES:
        for key, tensor in state_dict.items():
            if not key.endswith(down_suffix) or tensor.ndim != 2:
                continue
            prefix = key[: -len(down_suffix)]
            up_key = f"{prefix}{up_suffix}"
            up_tensor = state_dict.get(up_key)
            if up_tensor is not None and up_tensor.ndim == 2:
                pairs[prefix] = (key, up_key, down_suffix, up_suffix)
    return pairs


def common_pair_prefixes(
    young_pairs: dict[str, tuple[str, str, str, str]],
    old_pairs: dict[str, tuple[str, str, str, str]],
) -> list[str]:
    young_keys = set(young_pairs)
    old_keys = set(old_pairs)
    if young_keys != old_keys:
        raise ValueError(
            "Young and old LoRA layer sets differ: "
            f"missing_in_old={sorted(young_keys - old_keys)[:12]}, "
            f"missing_in_young={sorted(old_keys - young_keys)[:12]}"
        )
    return sorted(young_keys & old_keys)


def svdmix(M0: torch.Tensor, M1: torch.Tensor, alpha: float) -> torch.Tensor:
    if M0.shape != M1.shape:
        raise ValueError(f"svdmix shape mismatch: {tuple(M0.shape)} vs {tuple(M1.shape)}")
    original_dtype = M0.dtype
    U0, S0, Vh0 = torch.linalg.svd(M0.float(), full_matrices=False)
    U1, S1, Vh1 = torch.linalg.svd(M1.float(), full_matrices=False)
    fused = (alpha * U0 + (1.0 - alpha) * U1) @ torch.diag(
        alpha * S0 + (1.0 - alpha) * S1
    ) @ (alpha * Vh0 + (1.0 - alpha) * Vh1)
    return fused.to(original_dtype)


def save_fused_lora(tensors: dict[str, torch.Tensor], output_dir: Path, metadata: dict[str, str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / LORA_WEIGHT_NAME
    save_file(tensors, str(out_file), metadata=metadata)
    (output_dir / "svdmix_fuse_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_file


def get_lora_states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    global _YOUNG_STATE, _OLD_STATE
    if _YOUNG_STATE is None or _OLD_STATE is None:
        _YOUNG_STATE = dict(load_file(str(resolve_lora_path(CONFIG["young_lora"]))))
        _OLD_STATE = dict(load_file(str(resolve_lora_path(CONFIG["old_lora"]))))
    return _YOUNG_STATE, _OLD_STATE


def resolve_lora_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / LORA_WEIGHT_NAME
    if not resolved.is_file():
        raise FileNotFoundError(f"LoRA not found: {resolved}")
    return resolved


def load_image(path: str | Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize(
        (CONFIG["image_size"], CONFIG["image_size"]), Image.LANCZOS
    )


def save_like_source(image: Image.Image, input_path: str | Path, output_path: str | Path) -> None:
    source = ImageOps.exif_transpose(Image.open(input_path))
    output = image.convert("RGB").resize(source.size, Image.LANCZOS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs = {"quality": 95, "subsampling": 0}
    output.save(output_path, **save_kwargs)


def group_from_age(age: int | float | None) -> str:
    if age is None:
        return CONFIG["default_source_group"]
    value = int(round(float(age)))
    for group, data in load_json(CONFIG["coefficients_json"])["groups"].items():
        age_min = int(data["age_min"])
        age_max = data["age_max"]
        if value >= age_min and (age_max is None or value <= int(age_max)):
            return normalize_group(group)
    return "60_"


def normalize_group(group: str) -> str:
    group = str(group).strip()
    if group == "60":
        return "60_"
    if group not in GROUP_ORDER:
        raise ValueError(f"Unknown age group: {group}")
    return group


def group_anchor_age(group: str) -> int:
    return int(load_json(CONFIG["coefficients_json"])["groups"][normalize_group(group)]["anchor_age"])


def alpha_for_target(target_group: str) -> float:
    groups = load_json(CONFIG["coefficients_json"])["groups"]
    young_age = float(groups["20_24"]["anchor_age"])
    old_age = float(groups["60_"]["anchor_age"])
    target = float(groups[normalize_group(target_group)]["anchor_age"])
    alpha = 1.0 - ((target - young_age) / (old_age - young_age))
    return round(max(0.0, min(1.0, alpha)), 4)


def alpha_cache_key(alpha: float) -> str:
    return f"alpha_{format_float(round(float(alpha), 4))}"


def format_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _estimate_source_age(image_path: str) -> float | None:
    if not CONFIG["infer_source_age"]:
        return None
    age, _ = estimate_insightface_source_age(image_path, det_size=640)
    return age


def _resolve_output_path(image_path: str, output_path: str, output_dir: str | None) -> Path:
    if output_path:
        return Path(output_path)
    base = Path(output_dir or "/tmp/your-new-face/outputs")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{Path(image_path).stem}_svdmix{Path(image_path).suffix}"


def validate_paths(paths: dict[str, str]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing SVDMix model paths:\n  " + "\n  ".join(missing))


@lru_cache(maxsize=8)
def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
