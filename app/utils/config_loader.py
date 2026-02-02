from __future__ import annotations

from pathlib import Path
import os
import yaml

from app.configs.config_schema import InferenceConfig


def _resolve_path(p: str | Path, base_dir: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def load_config(yaml_path: str | Path) -> InferenceConfig:
    yaml_path = Path(yaml_path).expanduser().resolve()
    base_dir = yaml_path.parent

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Resolve paths before validation so Pydantic checks existence correctly
    for b_id, b in raw.get("buildings", {}).items():
        b["csv_path"] = _resolve_path(b["csv_path"], base_dir)

        for a_name, a in b.get("appliances", {}).items():
            a["classifier_path"] = _resolve_path(a["classifier_path"], base_dir)
            a["regressor_path"] = _resolve_path(a["regressor_path"], base_dir)

    return InferenceConfig.model_validate(raw)
