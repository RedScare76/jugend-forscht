from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .utils import load_yaml


class PipelineConfig(BaseModel):
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1"
    openai_org: Optional[str] = None
    openai_project: Optional[str] = None

    handwriting_api_url: str = "http://localhost:8000"
    handwriting_output_dir: str = "data/handwriting_out"
    handwriting_timeout_s: float = 120.0

    dataset_output_dir: str = "data/kraken_training"
    render_height: int = 80
    render_padding: int = 6
    bias_min: float = 0.2
    bias_max: float = 0.9
    normalization: str = "NFD"

    kraken_device: str = "cpu"
    kraken_epochs: int = 20
    kraken_partition: float = 0.9

    @classmethod
    def from_sources(cls, config_path: Optional[Path] = None) -> "PipelineConfig":
        data = {}
        data.update(load_yaml(config_path))
        env = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "openai_model": os.getenv("OPENAI_MODEL"),
            "openai_org": os.getenv("OPENAI_ORG"),
            "openai_project": os.getenv("OPENAI_PROJECT"),
            "handwriting_api_url": os.getenv("HANDWRITING_API_URL"),
            "handwriting_output_dir": os.getenv("HANDWRITING_OUTPUT_DIR"),
            "handwriting_timeout_s": os.getenv("HANDWRITING_TIMEOUT_S"),
            "dataset_output_dir": os.getenv("DATASET_OUTPUT_DIR"),
            "render_height": os.getenv("RENDER_HEIGHT"),
            "render_padding": os.getenv("RENDER_PADDING"),
            "bias_min": os.getenv("BIAS_MIN"),
            "bias_max": os.getenv("BIAS_MAX"),
            "normalization": os.getenv("GT_NORMALIZATION"),
            "kraken_device": os.getenv("KRAKEN_DEVICE"),
            "kraken_epochs": os.getenv("KRAKEN_EPOCHS"),
            "kraken_partition": os.getenv("KRAKEN_PARTITION"),
        }
        for key, value in env.items():
            if value is None or value == "":
                continue
            data[key] = value
        return cls(**_coerce_types(data))


def _coerce_types(data: dict) -> dict:
    casted = dict(data)
    for key in ["handwriting_timeout_s", "bias_min", "bias_max", "kraken_partition"]:
        if key in casted:
            casted[key] = float(casted[key])
    for key in ["render_height", "render_padding", "kraken_epochs"]:
        if key in casted:
            casted[key] = int(casted[key])
    return casted
