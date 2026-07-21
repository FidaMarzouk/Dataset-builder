from typing import Any, Callable, Dict, List, Optional, Union

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from .dataset_loader import load_custom_dataset, load_standard_dataset, limit_dataset
from .use_cases import UseCase, build_metrics, list_use_cases


class Evaluation:

    def __init__(
        self,
        model: Optional[Any] = None,
        default_dataset: Optional[EvaluationDataset] = None,
    ):
        self.model = model
        self._default_dataset = default_dataset

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


    @staticmethod
    def _build_test_cases(
        dataset: EvaluationDataset,
        #called once per golden to produce actual outputs
        actual_output_fn: Optional[Callable[[str], str]],
        #a given precomputed list
        actual_outputs: Optional[List[str]],
    ) -> List[LLMTestCase]:

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
                    f"({len(actual_outputs)} != {len(dataset.goldens)}). "
                    "Pass num_goldens=<len(actual_outputs)> to customeval() to "
                    "align the two, or trim actual_outputs to match the dataset."
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
        #ignored if metrics are passed explicitly 
        use_case: Optional[UseCase] = None,
        dataset: Optional[Union[EvaluationDataset, str]] = None,
        metrics: Optional[List[BaseMetric]] = None,
        actual_output_fn: Optional[Callable[[str], str]] = None,
        actual_outputs: Optional[List[str]] = None,
        num_goldens: Optional[int] = None,          
        shuffle_goldens: bool = False,              
        shuffle_seed: Optional[int] = None, 
        metric_overrides: Optional[Dict[str, Any]] = None,
        run_async: bool = True,
        show_indicator: bool = True,
        #pull the stored score/reason/pass-fail from its local cache file instead of calling the metric's judge model again
        #based on test case + metric config
        use_cache: bool = False,
        #a tag to distinguish runs
        identifier: Optional[str] = None,
    ):

        resolved_dataset = self._resolve_dataset(dataset)
        if num_goldens is not None:
            resolved_dataset = limit_dataset(
            resolved_dataset, num_goldens, shuffle=shuffle_goldens, seed=shuffle_seed
        )
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
