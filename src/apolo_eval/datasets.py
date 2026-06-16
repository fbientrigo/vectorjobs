"""Dataset records and JSONL loaders for job-understanding evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class JobPair:
    anchor: str
    candidate: str
    label: int
    difficulty: str


@dataclass(frozen=True)
class JobTriplet:
    anchor: str
    positive: str
    negative: str
    difficulty: str


def load_pairs(path: str | Path) -> list[JobPair]:
    rows = _read_jsonl(path)
    return [
        JobPair(
            anchor=str(row["anchor"]),
            candidate=str(row["candidate"]),
            label=int(row["label"]),
            difficulty=str(row["difficulty"]),
        )
        for row in rows
    ]


def load_triplets(path: str | Path) -> list[JobTriplet]:
    rows = _read_jsonl(path)
    return [
        JobTriplet(
            anchor=str(row["anchor"]),
            positive=str(row["positive"]),
            negative=str(row["negative"]),
            difficulty=str(row["difficulty"]),
        )
        for row in rows
    ]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_no} is not a JSON object")
            records.append(record)
    return records
