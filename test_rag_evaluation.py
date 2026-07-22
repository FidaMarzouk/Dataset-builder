import os
from dotenv import load_dotenv

from evaluation import Evaluation, UseCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from app_clients.rag_pgvector_client import rag_pgvector

load_dotenv()

judge_model = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")
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
)

print(results)