# adapters/halubench_adapter.py — PASS only, FAIL rows never generated
from datasets import load_dataset
from ..common import make_id

def load_halubench(split="test"):
    return load_dataset("PatronusAI/HaluBench", split=split)

def halubench_to_golden_drafts(ds, n_samples: int):
    pass_rows = ds.filter(lambda r: r["label"] == "PASS").shuffle(seed=42).select(range(n_samples))
    drafts = []
    for row in pass_rows:
        drafts.append({
            "input": row["question"],
            "expected_output": row["answer"],
            "context": [row["passage"]],
            "retrieval_context": [row["passage"]],
            "additional_metadata": {
                "task_type": "rag_qa", "priority": "P2",
                "source": "seed", "ci_stage": "nightly", "linked_ticket": None,
            },
            "custom_column_key_values": {
                "origin_dataset": "halubench",
                "origin_source_ds": row["source_ds"],
                "halubench_label": "PASS",
                "origin_id": make_id("halubench", row["question"]),
            },
            "comments": "Seeded from HaluBench, verified-faithful answer only",
        })
    return drafts