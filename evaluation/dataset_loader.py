"""
dataset_loader.py

Loads Goldens into a DeepEval EvaluationDataset, either from the
project's standard dataset (standard_dataset/goldens.jsonl) or from any
custom .jsonl file following the same schema:

    {
      "input": str,
      "expected_output": str,
      "context": [str, ...],
      "retrieval_context": [str, ...],
      "additional_metadata": {...},
      "custom_column_key_values": {...},
      "comments": str
    }
"""

import json
from pathlib import Path
from typing import List, Optional

from deepeval.dataset import EvaluationDataset, Golden

# Adjust this if you move evaluation/ relative to standard_dataset/.
# Currently assumes:
#   ELYAEVAL/
#     standard_dataset/goldens.jsonl
#     evaluation/dataset_loader.py   <- this file
DEFAULT_STANDARD_DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "standard_dataset" / "goldens.jsonl"
)


def _read_goldens_jsonl(path: Path) -> List[Golden]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    goldens: List[Golden] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from e

            goldens.append(
                Golden(
                    input=record["input"],
                    expected_output=record.get("expected_output"),
                    context=record.get("context"),
                    retrieval_context=record.get("retrieval_context"),
                    additional_metadata=record.get("additional_metadata"),
                    comments=record.get("comments"),
                    custom_column_key_values=record.get("custom_column_key_values"),
                )
            )
    return goldens


def load_standard_dataset(path: Optional[Path] = None) -> EvaluationDataset:
    """Loads the project's standard goldens.jsonl into an EvaluationDataset."""
    dataset_path = Path(path) if path else DEFAULT_STANDARD_DATASET_PATH
    goldens = _read_goldens_jsonl(dataset_path)
    return EvaluationDataset(goldens=goldens)


def load_custom_dataset(path: str) -> EvaluationDataset:
    """Loads any .jsonl file following the same golden schema."""
    goldens = _read_goldens_jsonl(Path(path))
    return EvaluationDataset(goldens=goldens)


def filter_by_metadata(dataset: EvaluationDataset, **filters) -> EvaluationDataset:
    """
    Convenience helper: returns a new EvaluationDataset containing only
    goldens whose additional_metadata matches every key/value in filters.

    e.g. filter_by_metadata(ds, ci_stage="nightly", priority="P2")
    """
    filtered = [
        g
        for g in dataset.goldens
        if g.additional_metadata
        and all(g.additional_metadata.get(k) == v for k, v in filters.items())
    ]
    return EvaluationDataset(goldens=filtered)
