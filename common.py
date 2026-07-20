import json, hashlib
from pydantic import BaseModel, ValidationError
from typing import Optional, List

class GoldenMetadata(BaseModel):
    task_type: str
    priority: str
    source: str
    ci_stage: str

class GoldenDraft(BaseModel):
    input: str
    expected_output: str
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    additional_metadata: GoldenMetadata
    custom_column_key_values: dict
    comments: Optional[str] = None

def make_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode()).hexdigest()[:16]

def write_to_standard_dataset(drafts: list[dict], path: str):
    existing_ids = set()
    try:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                existing_ids.add(row["custom_column_key_values"].get("origin_id"))
    except FileNotFoundError:
        pass

    validated, skipped_dupes = [], 0
    for d in drafts:
        try:
            draft = GoldenDraft(**d).model_dump()
        except ValidationError as e:
            print(f"Skipping malformed draft: {e}")
            continue
        if draft["custom_column_key_values"].get("origin_id") in existing_ids:
            skipped_dupes += 1
            continue
        validated.append(draft)
        existing_ids.add(draft["custom_column_key_values"]["origin_id"])

    with open(path, "a") as f:
        for d in validated:
            f.write(json.dumps(d) + "\n")
    print(f"Wrote {len(validated)} new goldens, skipped {skipped_dupes} duplicates, to {path}")