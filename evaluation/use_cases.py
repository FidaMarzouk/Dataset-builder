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
    #optional if we want to override the thresshold of a specific metric 
    #either that or updating the initial config
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Any]:
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
