"""
use_cases.py

Central registry mapping evaluation use cases to the DeepEval metric
classes considered "standard" for that use case.

Add a new use case by:
  1. Adding a member to `UseCase`.
  2. Adding an entry to `USE_CASE_METRICS` with the metric classes and
     their default constructor kwargs.

We start with 4 use cases, chosen to match what the standard
goldens.jsonl schema already supports (input / expected_output /
context / retrieval_context):

  RAG            -> end-to-end generation quality over retrieved context
  RETRIEVAL      -> quality of the retriever alone, independent of generation
  SUMMARIZATION  -> faithfulness/coverage of a generated summary
  QA_CORRECTNESS -> general grounded Q&A correctness, no retrieval context required

NOTE: metrics classes/params below reflect the DeepEval metrics API.
Verify class names/kwargs against your installed `deepeval` version
(`pip show deepeval`) before first run -- the library evolves quickly.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    SummarizationMetric,
    HallucinationMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams


class UseCase(str, Enum):
    RAG = "rag"
    RETRIEVAL = "retrieval"
    SUMMARIZATION = "summarization"
    QA_CORRECTNESS = "qa_correctness"


# Each spec is {"class": <MetricClass>, "kwargs": {...}}.
# Kept as class + kwargs (not instances) so build_metrics() can hand
# back a *fresh* instance every call -- DeepEval metric instances are
# stateful once .measure() has run, so reuse across evals is unsafe.
USE_CASE_METRICS: Dict[UseCase, List[Dict[str, Any]]] = {
    UseCase.RAG: [
        {"class": FaithfulnessMetric, "kwargs": {"threshold": 0.7}},
        {"class": AnswerRelevancyMetric, "kwargs": {"threshold": 0.7}},
        {"class": ContextualPrecisionMetric, "kwargs": {"threshold": 0.7}},
        {"class": ContextualRecallMetric, "kwargs": {"threshold": 0.7}},
    ],
    UseCase.RETRIEVAL: [
        {"class": ContextualPrecisionMetric, "kwargs": {"threshold": 0.7}},
        {"class": ContextualRecallMetric, "kwargs": {"threshold": 0.7}},
        {"class": ContextualRelevancyMetric, "kwargs": {"threshold": 0.7}},
    ],
    UseCase.SUMMARIZATION: [
        {"class": SummarizationMetric, "kwargs": {"threshold": 0.7}},
    ],
    UseCase.QA_CORRECTNESS: [
        {"class": HallucinationMetric, "kwargs": {"threshold": 0.5}},
        {
            "class": GEval,
            "kwargs": {
                "name": "Correctness",
                "criteria": (
                    "Determine whether the actual output is factually "
                    "correct and semantically equivalent to the expected "
                    "output, given the input."
                ),
                "evaluation_params": [
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                "threshold": 0.7,
            },
        },
    ],
}


def list_use_cases() -> List[str]:
    """Returns every registered use case value, e.g. for CLI help text."""
    return [uc.value for uc in UseCase]


def get_default_metric_specs(use_case: UseCase) -> List[Dict[str, Any]]:
    """Returns the raw {class, kwargs} specs registered for a use case."""
    if use_case not in USE_CASE_METRICS:
        raise ValueError(
            f"Unknown use case '{use_case}'. Available: {list_use_cases()}"
        )
    return USE_CASE_METRICS[use_case]


def build_metrics(
    use_case: UseCase,
    model: Optional[Any] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """
    Instantiates the metric objects registered for a use case.

    model:      optional deepeval-compatible LLM (e.g. your local Ollama
                judge wrapper) injected into every metric that doesn't
                already specify its own `model` kwarg.
    overrides:  optional {metric_class_name: kwargs} dict to override or
                extend the default kwargs for one specific metric, e.g.
                overrides={"FaithfulnessMetric": {"threshold": 0.9}}.
    """
    overrides = overrides or {}
    specs = get_default_metric_specs(use_case)
    metrics = []
    for spec in specs:
        cls = spec["class"]
        kwargs = dict(spec["kwargs"])
        kwargs.update(overrides.get(cls.__name__, {}))
        if model is not None:
            kwargs.setdefault("model", model)
        metrics.append(cls(**kwargs))
    return metrics
