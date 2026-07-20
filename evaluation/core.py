"""
core.py

Evaluation: the main entry point of the framework. Wraps DeepEval's
evaluate() with project conventions -- standard dataset by default,
use-case-to-metric registry, and consistent result handling -- so day
to day usage is a single call to customeval().
"""

from typing import Any, Callable, Dict, List, Optional, Union

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from .dataset_loader import load_custom_dataset, load_standard_dataset
from .use_cases import UseCase, build_metrics, list_use_cases


class Evaluation:
    """
    model:            optional DeepEval-compatible LLM (e.g. a wrapper
                       around your local Ollama judge) injected into every
                       metric that isn't given its own model explicitly.
    default_dataset:  optional pre-loaded EvaluationDataset used whenever
                       customeval() isn't given one explicitly. Lazily
                       loads the standard dataset the first time it's
                       needed if you don't pass one.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        default_dataset: Optional[EvaluationDataset] = None,
    ):
        self.model = model
        self._default_dataset = default_dataset

    # ---- dataset helpers -------------------------------------------------

    def get_standard_dataset(self) -> EvaluationDataset:
        if self._default_dataset is None:
            self._default_dataset = load_standard_dataset()
        return self._default_dataset

    def get_custom_dataset(self, path: str) -> EvaluationDataset:
        return load_custom_dataset(path)

    def _resolve_dataset(
        self, dataset: Optional[Union[EvaluationDataset, str]]
    ) -> EvaluationDataset:
        if dataset is None:
            return self.get_standard_dataset()
        if isinstance(dataset, str):
            return self.get_custom_dataset(dataset)
        return dataset

    # ---- metric helpers ----------------------------------------------

    def _resolve_metrics(
        self,
        use_case: Optional[UseCase],
        metrics: Optional[List[BaseMetric]],
        metric_overrides: Optional[Dict[str, Any]],
    ) -> List[BaseMetric]:
        if metrics:
            return metrics
        if use_case is None:
            raise ValueError(
                "Either 'use_case' or an explicit 'metrics' list is required."
            )
        return build_metrics(use_case, model=self.model, overrides=metric_overrides)

    # ---- goldens -> test cases --------------------------------------

    @staticmethod
    def _build_test_cases(
        dataset: EvaluationDataset,
        actual_output_fn: Optional[Callable[[str], str]],
        actual_outputs: Optional[List[str]],
    ) -> List[LLMTestCase]:
        """
        Populates dataset.test_cases from dataset.goldens and returns them.

        Exactly one of the following must be true:
          - actual_output_fn is given: called once per golden.input to
            produce actual_output (this is the "run my app under test"
            path).
          - actual_outputs is given: a pre-computed list aligned 1:1
            with dataset.goldens (useful if you generated outputs in a
            separate batch step, e.g. to control concurrency/cost).
          - dataset.test_cases is already populated upstream.
        """
        if actual_output_fn is not None:
            for golden in dataset.goldens:
                dataset.add_test_case(
                    LLMTestCase(
                        input=golden.input,
                        actual_output=actual_output_fn(golden.input),
                        expected_output=golden.expected_output,
                        context=golden.context,
                        retrieval_context=golden.retrieval_context,
                    )
                )
            return dataset.test_cases

        if actual_outputs is not None:
            if len(actual_outputs) != len(dataset.goldens):
                raise ValueError(
                    "actual_outputs length must match number of goldens "
                    f"({len(actual_outputs)} != {len(dataset.goldens)})."
                )
            for golden, output in zip(dataset.goldens, actual_outputs):
                dataset.add_test_case(
                    LLMTestCase(
                        input=golden.input,
                        actual_output=output,
                        expected_output=golden.expected_output,
                        context=golden.context,
                        retrieval_context=golden.retrieval_context,
                    )
                )
            return dataset.test_cases

        if dataset.test_cases:
            return dataset.test_cases

        raise ValueError(
            "No actual outputs available. Pass 'actual_output_fn' or "
            "'actual_outputs' to customeval(), or populate "
            "dataset.test_cases beforehand via dataset.add_test_case()."
        )

    # ---- main entry point ----------------------------------------------

    def customeval(
        self,
        use_case: Optional[UseCase] = None,
        dataset: Optional[Union[EvaluationDataset, str]] = None,
        metrics: Optional[List[BaseMetric]] = None,
        actual_output_fn: Optional[Callable[[str], str]] = None,
        actual_outputs: Optional[List[str]] = None,
        metric_overrides: Optional[Dict[str, Any]] = None,
        run_async: bool = True,
        show_indicator: bool = True,
        use_cache: bool = False,
        identifier: Optional[str] = None,
    ):
        """
        Runs a DeepEval evaluation end to end.

        use_case:          one of UseCase.* -- pulls the registered default
                            metrics for that use case. Ignored if 'metrics'
                            is passed explicitly.
        dataset:            EvaluationDataset, path to a custom .jsonl, or
                            None (falls back to the standard dataset).
        metrics:            explicit metric instances; overrides use_case.
        actual_output_fn:   fn(input_str) -> actual_output_str; called once
                            per golden to run your app under test.
        actual_outputs:     pre-computed outputs aligned 1:1 with
                            dataset.goldens, if you generated them upstream.
        metric_overrides:   per-metric kwargs overrides keyed by class name,
                            applied only when metrics are built from use_case.
        run_async:          evaluate test cases concurrently (DeepEval
                            default is True; matches project convention).
        use_cache:          reuse cached metric scores for identical
                            test-case/metric pairs where available.
        identifier:         optional run label, useful for CI logs / cost
                            tracking.

        Returns whatever deepeval.evaluate() returns (an EvaluationResult
        containing per-test-case, per-metric scores/reasons/pass-fail).
        """
        resolved_dataset = self._resolve_dataset(dataset)
        resolved_metrics = self._resolve_metrics(use_case, metrics, metric_overrides)
        test_cases = self._build_test_cases(
            resolved_dataset, actual_output_fn, actual_outputs
        )

        return evaluate(
            test_cases=test_cases,
            metrics=resolved_metrics,
            async_config=AsyncConfig(run_async=run_async),
            display_config=DisplayConfig(
                show_indicator=show_indicator, print_results=True
            ),
            cache_config=CacheConfig(use_cache=use_cache),
            identifier=identifier,
        )

    # ---- convenience ----------------------------------------------------

    @staticmethod
    def available_use_cases() -> List[str]:
        return list_use_cases()
