from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from new_face.svdmix_age_task import (
    CONFIG,
    LORA_WEIGHT_NAME,
    age_grid_values,
    alpha_for_age,
    get_lora_states,
    precomputed_lora_dir_for_age,
    resolve_lora_path,
    fuse_lora_state_dict,
    save_fused_lora,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute AgeBooth-style SVDMix LoRA files for the age grid."
    )
    parser.add_argument("--output-dir", default=CONFIG["precomputed_lora_dir"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CONFIG["precomputed_lora_dir"] = args.output_dir

    young_path = resolve_lora_path(CONFIG["young_lora"])
    old_path = resolve_lora_path(CONFIG["old_lora"])
    young_state, old_state = get_lora_states()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "young_lora": str(young_path),
        "old_lora": str(old_path),
        "age_grid": [],
    }

    for age in age_grid_values():
        alpha = alpha_for_age(age)
        out_dir = precomputed_lora_dir_for_age(age)
        out_file = out_dir / LORA_WEIGHT_NAME
        if out_file.is_file() and not args.force:
            print(f"[SKIP] age={age:g} alpha={alpha:.4f}: {out_file}")
        else:
            print(f"[FUSE] age={age:g} alpha={alpha:.4f}: {out_file}")
            fused_state = fuse_lora_state_dict(young_state, old_state, alpha)
            save_fused_lora(
                fused_state,
                out_dir,
                {
                    "agebooth.style": "svdmix_per_factor",
                    "agebooth.alpha_young": f"{alpha:.6f}",
                    "agebooth.grid_age": f"{age:.4f}",
                    "agebooth.young_lora": str(young_path),
                    "agebooth.old_lora": str(old_path),
                },
            )
        manifest["age_grid"].append(
            {
                "age": age,
                "alpha_young": alpha,
                "path": str(out_file),
            }
        )

    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] manifest: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
