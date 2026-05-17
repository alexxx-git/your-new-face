"""
Загрузка промптов по возрасту (JSON на каждый age) и коэффициентов перехода
(исходный возраст → целевой, пол). Используется test_single_pass и др.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_COEF_KEYS = frozenset(
    {
        "strength",
        "cfg",
        "ip_adapter_scale",
        "custom_lora_scale",
        "age_slider_scale",
        "steps",
    }
)


def _substitute_placeholders(text: str, age: int, gender: str) -> str:
    return (
        text.replace("{age}", str(int(age)))
        .replace("{gender}", gender)
        .replace("{Age}", str(int(age)))
    )


def load_age_prompt_pair(
    prompts_dir: str | Path,
    age: int,
    gender: str,
) -> tuple[str, str]:
    """
    Читает ``prompts_dir / f"{age}.json"``. Поля: positive, negative (строки, опционально {age}, {gender}).
    Если файла нет — ValueError с подсказкой.
    """
    base = Path(prompts_dir)
    path = base / f"{int(age)}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Нет промпта для возраста {age}: ожидался файл {path}. "
            f"Создай JSON с полями positive и negative."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    pos = data.get("positive")
    neg = data.get("negative")
    if not isinstance(pos, str) or not isinstance(neg, str):
        raise ValueError(f"{path}: нужны строковые поля positive и negative")
    return _substitute_placeholders(pos, age, gender), _substitute_placeholders(
        neg, age, gender
    )


def load_age_prompt_pair_or_fallback(
    prompts_dir: str | Path | None,
    age: int,
    gender: str,
    fallback_positive,
    fallback_negative,
) -> tuple[str, str]:
    if not prompts_dir:
        return fallback_positive(age, gender), fallback_negative(age)
    try:
        return load_age_prompt_pair(prompts_dir, age, gender)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[WARN] age_prompts: {exc} — используется встроенный fallback.")
        return fallback_positive(age, gender), fallback_negative(age)


def _gender_matches(rule_genders: list[str] | None, gender: str) -> bool:
    if not rule_genders:
        return True
    g = gender.lower().strip()
    allowed = {x.lower() for x in rule_genders}
    return g in allowed or "any" in allowed


def resolve_age_coefficients(
    coefficients_path: str | Path | None,
    *,
    source_age: int | None,
    target_age: int,
    gender: str,
) -> dict[str, Any]:
    """
    Подбирает коэффициенты из JSON. Правила ``transitions`` — по порядку, первое совпадение
    перезаписывает ``defaults``.

    Поля правила (все опционально, кроме совпадения по возрастам):
      - source_age_min, source_age_max — если задан ``source_age``, он должен попадать в интервал
      - require_source_age: true — правило применяется только если ``source_age`` не None
      - target_age_min, target_age_max
      - genders: ["male", "female", "any"]
      - overrides: подмножество _COEF_KEYS
    """
    if not coefficients_path:
        return {}
    path = Path(coefficients_path)
    if not path.is_file():
        print(f"[WARN] age_coefficients: файл не найден {path}, коэффициенты из CONFIG.")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[WARN] age_coefficients: {exc}")
        return {}

    out: dict[str, Any] = dict(data.get("defaults") or {})
    for key in list(out.keys()):
        if key not in _COEF_KEYS:
            del out[key]

    for rule in data.get("transitions", []):
        if not _gender_matches(rule.get("genders"), gender):
            continue

        t_min = int(rule.get("target_age_min", 0))
        t_max = int(rule.get("target_age_max", 150))
        if not (t_min <= int(target_age) <= t_max):
            continue

        has_source_window = "source_age_min" in rule or "source_age_max" in rule
        sa_min = int(rule.get("source_age_min", 0))
        sa_max = int(rule.get("source_age_max", 150))
        if has_source_window:
            if source_age is None:
                continue
            if not (sa_min <= int(source_age) <= sa_max):
                continue

        ov = rule.get("overrides") or {}
        for k, v in ov.items():
            if k in _COEF_KEYS:
                out[k] = v
        break

    return out


def merge_run_config(base: dict[str, Any], coef: dict[str, Any]) -> dict[str, Any]:
    """Поверх CONFIG (или копии) накладывает только разрешённые коэффициенты."""
    merged = dict(base)
    for k, v in coef.items():
        if k in _COEF_KEYS:
            merged[k] = v
    return merged
