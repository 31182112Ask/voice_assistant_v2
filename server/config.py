"""配置加載 — 將 config.yaml 解析為帶屬性訪問的命名空間。"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def load_config(path: str | pathlib.Path | None = None) -> SimpleNamespace:
    cfg_path = pathlib.Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _ns(raw)
    cfg._root = ROOT
    return cfg
