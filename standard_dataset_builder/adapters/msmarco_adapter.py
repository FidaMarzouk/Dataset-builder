from datasets import load_dataset
from ..common import make_id,sample_streaming

def load_msmarco(split="test"):
    return load_dataset("microsoft/ms_marco", "v2.1", split=split, streaming=True)

def msmarco_to_golden_drafts(ds, n_samples: int):
    sample = sample_streaming(ds, n_samples)
    drafts = []
    for row in sample:
        if not row["answers"]:
            continue
        selected = [p for p, sel in zip(row["passages"]["passage_text"], row["passages"]["is_selected"]) if sel == 1]
        drafts.append({
            "input": row["query"],
            "expected_output": row["answers"][0],
            "context": selected or row["passages"]["passage_text"],
            "retrieval_context": selected or row["passages"]["passage_text"],
            "additional_metadata": {
                "task_type": "rag_qa", "priority": "P2",
                "source": "seed", "ci_stage": "nightly", "linked_ticket": None,
            },
            "custom_column_key_values": {
                "origin_dataset": "ms_marco/v2.1",
                "origin_id": make_id("msmarco", row["query"]),
            },
            "comments": "Seeded from MS MARCO v2.1",
        })
    return drafts