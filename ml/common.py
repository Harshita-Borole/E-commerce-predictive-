from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "ml" / "models"
REPORTS_DIR = ROOT / "docs" / "ml_reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    return logging.getLogger(name)


def load(table: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / f"{table}.csv", low_memory=False)


def save_report(name: str, report: dict) -> Path:
    path = REPORTS_DIR / f"{name}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def print_metric_block(title: str, metrics: dict) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:.4f}")
        else:
            print(f"  {k:<28} {v}")
