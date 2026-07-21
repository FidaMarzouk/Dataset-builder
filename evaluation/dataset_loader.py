import json
from pathlib import Path
from typing import List, Optional
import random

from deepeval.dataset import EvaluationDataset, Golden

DEFAULT_STANDARD_DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "standard_dataset" / "goldens.jsonl"
)

#reads jsonl objects converts them into a list of golden objects
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

#wraps the golden objects in an evaluation dataset
def load_standard_dataset(path: Optional[Path] = None) -> EvaluationDataset:
    """Loads the project's standard goldens.jsonl into an EvaluationDataset."""
    dataset_path = Path(path) if path else DEFAULT_STANDARD_DATASET_PATH
    goldens = _read_goldens_jsonl(dataset_path)
    return EvaluationDataset(goldens=goldens)

#loads any jsonl file following the same schema 
def load_custom_dataset(path: str) -> EvaluationDataset:
    goldens = _read_goldens_jsonl(Path(path))
    return EvaluationDataset(goldens=goldens)


def filter_by_metadata(dataset: EvaluationDataset, **filters) -> EvaluationDataset:
    filtered = [
        g
        for g in dataset.goldens
        if g.additional_metadata
        and all(g.additional_metadata.get(k) == v for k, v in filters.items())
    ]
    return EvaluationDataset(goldens=filtered)

def limit_dataset(
    dataset: EvaluationDataset,
    n: int,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> EvaluationDataset:
    goldens = list(dataset.goldens)
    if shuffle:
        random.Random(seed).shuffle(goldens)
    return EvaluationDataset(goldens=goldens[:n])