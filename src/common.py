"""Config loading + incremental JSONL helpers shared by all stages."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike = "configs/exp1.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(p)
    return cfg


def resolve(path: str | os.PathLike) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    p = resolve(path)
    if not p.exists():
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class JsonlWriter:
    """Append-only writer that flushes+fsyncs every record (crash-safe resume)."""

    def __init__(self, path: str | os.PathLike):
        self.path = resolve(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, rec: dict) -> None:
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_done_keys(path: str | os.PathLike, key: str) -> set:
    return {r[key] for r in read_jsonl(path) if key in r}
