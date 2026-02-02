from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

Unit = Literal["w", "kw"]


class ApplianceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_size: int = Field(..., description="Must match training window size (odd integer).")
    clf_threshold: float = Field(0.5, ge=0.0, le=1.0)
    on_w: float = Field(50.0, ge=0.0)

    classifier_path: Path
    regressor_path: Path
    regressor_output_unit: Unit = "kw"

    # Optional per-appliance overrides (inherit from building if None)
    rows_per_hour: Optional[int] = Field(None, gt=0)
    max_mains_w: Optional[float] = Field(None, gt=0)

    @field_validator("window_size")
    @classmethod
    def window_size_must_be_positive_odd(cls, v: int) -> int:
        if v <= 0 or v % 2 == 0:
            raise ValueError("window_size must be a positive odd integer (e.g., 181, 301).")
        return v

    @field_validator("classifier_path", "regressor_path")
    @classmethod
    def model_paths_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Model path does not exist: {v}")
        return v


class BuildingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_path: Path

    # Defaults for all appliances in this building (can be overridden per appliance)
    rows_per_hour: int = Field(5000, gt=0)
    max_mains_w: Optional[float] = Field(None, gt=0)

    appliances: Dict[str, ApplianceConfig]

    @field_validator("csv_path")
    @classmethod
    def csv_path_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"CSV path does not exist: {v}")
        return v

    @model_validator(mode="after")
    def ensure_appliances_non_empty(self) -> "BuildingConfig":
        if not self.appliances:
            raise ValueError("Each building must define at least one appliance under 'appliances'.")
        return self


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    buildings: Dict[str, BuildingConfig]

    @model_validator(mode="after")
    def ensure_buildings_non_empty(self) -> "InferenceConfig":
        if not self.buildings:
            raise ValueError("Config must define at least one building under 'buildings'.")
        return self
