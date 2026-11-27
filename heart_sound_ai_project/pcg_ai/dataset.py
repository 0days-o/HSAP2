"""Utilities for reading the PhysioNet/CinC Challenge 2016 labels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Record:
    record_id: str
    path: Path
    label: int  # 0 = reference normal; 1 = reference abnormal
    source: str

    @property
    def unique_id(self) -> str:
        return f"{self.source}/{self.record_id}"


def _parse_reference_line(line: str) -> tuple[str, int] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = re.split(r"[\s,;|]+", line)
    if len(parts) < 2:
        return None
    try:
        label_raw = int(float(parts[1]))
    except ValueError:
        return None  # skip header or malformed line
    if label_raw == -1:
        return parts[0], 0
    if label_raw == 1:
        return parts[0], 1
    return None


def load_records(dataset_root: str | Path) -> list[Record]:
    """Find WAVs referenced by REFERENCE.csv files under dataset_root.

    The Challenge labels are -1 for normal and +1 for abnormal. Files without a
    matching WAV or a usable label are skipped with a warning.
    """
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    references = sorted(root.rglob("REFERENCE.csv"))
    if not references:
        raise FileNotFoundError(
            f"No REFERENCE.csv found under {root}. Extract the PhysioNet training ZIP there first."
        )

    result: list[Record] = []
    seen: set[tuple[Path, str]] = set()
    for ref_path in references:
        source = ref_path.parent.name
        try:
            lines = ref_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            print(f"WARNING: cannot read {ref_path}: {exc}")
            continue

        for line in lines:
            parsed = _parse_reference_line(line)
            if parsed is None:
                continue
            record_name, label = parsed
            # Most PhysioNet archives keep the WAV next to its REFERENCE.csv.
            candidate = ref_path.parent / f"{record_name}.wav"
            if not candidate.exists():
                candidates = list(ref_path.parent.rglob(f"{record_name}.wav"))
                if not candidates:
                    print(f"WARNING: missing WAV for {record_name!r} listed in {ref_path}")
                    continue
                candidate = candidates[0]
            key = (candidate.resolve(), record_name)
            if key in seen:
                continue
            seen.add(key)
            result.append(Record(record_name, candidate.resolve(), label, source))

    if not result:
        raise RuntimeError(f"No labeled WAV records found in {root}")
    result.sort(key=lambda r: (r.source, r.record_id))
    return result
