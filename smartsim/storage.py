"""Historial de evaluaciones: guarda cada punto evaluado por el simulador real."""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple


class HistoryStore:
    def __init__(self, param_names: List[str]):
        self.param_names = param_names
        self.rows: List[dict] = []

    def add(self, iteration: int, params: Dict[str, float], result: float, phase: str,
            acquisition_score: float = None) -> None:
        row = {
            "iteration": iteration,
            **{name: params[name] for name in self.param_names},
            "result": result,
            "phase": phase,
            "acquisition_score": acquisition_score,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }
        self.rows.append(row)

    def as_history_tuples(self) -> List[Tuple[Dict[str, float], float]]:
        return [({k: r[k] for k in self.param_names}, r["result"]) for r in self.rows]

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["iteration", *self.param_names, "result", "phase", "acquisition_score", "timestamp"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def __len__(self) -> int:
        return len(self.rows)
