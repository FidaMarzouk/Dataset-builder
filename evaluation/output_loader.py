import csv
import json
from pathlib import Path
from typing import Dict

def load_precomputed_outputs(path: str, key_field: str = "input") -> Dict[str, dict]:
    p = Path(path)
    outputs: Dict[str, dict] = {}

    if p.suffix.lower() == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "retrieval_context" in row and row["retrieval_context"]:
                    row["retrieval_context"] = row["retrieval_context"].split("|||")
                outputs[row[key_field]] = row
    else:  # jsonl
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                outputs[rec[key_field]] = rec

    return outputs