from datasets import load_dataset
from ..common import make_id,sample_streaming

def load_halubench(split="test"):
    return load_dataset("PatronusAI/HaluBench", split=split, streaming=True)

def halubench_to_golden_drafts(ds, n_samples: int):
    pass_only = ds.filter(lambda r: r["label"] == "PASS")
    sample = sample_streaming(pass_only, n_samples)
    drafts = []
    for row in sample:
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