from datasets import load_dataset
from ..common import make_id

def load_ragbench_subset(subset_name: str, split="test"):
    return load_dataset("rungalileo/ragbench", subset_name, split=split)

def ragbench_to_golden_drafts(subset_name: str, ds, n_samples: int):
    sample = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    drafts = []
    for row in sample:
        docs = row["documents"] if isinstance(row["documents"], list) else [row["documents"]]
        drafts.append({
            "input": row["question"],
            "expected_output": row["response"],
            "context": docs,
            "retrieval_context": docs,
            "additional_metadata": {
                "task_type": "rag_qa", "priority": "P2",
                "source": "seed", "ci_stage": "nightly", "linked_ticket": None,
            },
            "custom_column_key_values": {
                "origin_dataset": f"ragbench/{subset_name}",
                "origin_id": make_id("ragbench", subset_name, row["question"]),
            },
            "comments": f"Seeded from RAGBench subset {subset_name}",
        })
    return drafts