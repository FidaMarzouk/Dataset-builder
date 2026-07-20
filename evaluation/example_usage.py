"""
example_usage.py

Three common ways to call Evaluation.customeval(). Run with:
    python -m evaluation.example_usage
"""

from evaluation import Evaluation, UseCase


def my_rag_app(question: str) -> str:
    """Stand-in for your actual RAG pipeline call."""
    return f"<generated answer for: {question}>"


if __name__ == "__main__":
    ev = Evaluation(model=None)  # pass your Ollama/judge wrapper here

    # 1) Standard dataset + registered RAG metrics, running your app live
    results = ev.customeval(
        use_case=UseCase.RAG,
        actual_output_fn=my_rag_app,
    )

    # 2) Custom dataset file + retrieval-only metrics
    # results = ev.customeval(
    #     use_case=UseCase.RETRIEVAL,
    #     dataset="path/to/custom_goldens.jsonl",
    #     actual_output_fn=my_rag_app,
    # )

    # 3) Fully explicit metrics, bypassing the use-case registry
    # from deepeval.metrics import AnswerRelevancyMetric
    # results = ev.customeval(
    #     metrics=[AnswerRelevancyMetric(threshold=0.8)],
    #     actual_output_fn=my_rag_app,
    # )

    print(results)
