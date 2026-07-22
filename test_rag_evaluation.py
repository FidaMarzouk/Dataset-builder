import os
from dotenv import load_dotenv

from evaluation import Evaluation, UseCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from app_clients.rag_pgvector_client import rag_pgvector
from deepeval.models import OllamaModel

load_dotenv()

judge_model = OllamaModel(
    model=os.getenv("OLLAMA_LLM_MODEL", "llama3.1"),
    base_url="http://localhost:11434",
    #generation_kwargs={"response_format": {"type": "json_object"}},
    temperature=0,
)
ev = Evaluation(model=judge_model)

results = ev.customeval(
    use_case=UseCase.RAG,
    metrics=[
        FaithfulnessMetric(threshold=0.7, model=judge_model),
        AnswerRelevancyMetric(threshold=0.7, model=judge_model),
    ],
    actual_output_fn=rag_pgvector,
    num_goldens=5,
    shuffle_goldens=True,
    shuffle_seed=42,
    print_results=False,
)

for r in results.test_results:
    print(f"\nInput: {r.input}")
    print(f"Generated answer: {r.actual_output}")
    print(f"Expected output:  {r.expected_output}")
    print("Retrieved chunks:")
    for i, chunk in enumerate(r.retrieval_context or [], 1):
        print(f"  [{i}] {chunk[:200]}...")
    for m in r.metrics_data:
        print(f"  {m.name}: score={m.score} success={m.success}")
        print(f"    reason: {m.reason}")