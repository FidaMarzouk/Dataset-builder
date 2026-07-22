from typing import Any, Callable, Dict, List, Optional, Union
import inspect
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
    def _call_output_fn(fn, input_text, required_fields):
        sig = inspect.signature(fn)
        if "required_fields" in sig.parameters:
            result = fn(input_text, required_fields=required_fields)
        else:
            result = fn(input_text)
        return result if isinstance(result, dict) else {"actual_output": result}

    @staticmethod
    def _build_test_cases(dataset, actual_output_fn, actual_outputs,
                           precomputed_by_input=None, required_fields=None):
        required_fields = required_fields or {"input", "actual_output"}

        if actual_output_fn is not None:
            for golden in dataset.goldens:
                fields = Evaluation._call_output_fn(actual_output_fn, golden.input, required_fields)
                dataset.add_test_case(LLMTestCase(
                    input=golden.input,
                    actual_output=fields.get("actual_output"),
                    expected_output=golden.expected_output,
                    context=fields.get("context", golden.context),
                    retrieval_context=fields.get("retrieval_context", golden.retrieval_context),
                ))
            return dataset.test_cases

        if precomputed_by_input is not None:
            for golden in dataset.goldens:
                rec = precomputed_by_input.get(golden.input)
                if rec is None:
                    raise ValueError(f"No precomputed output found for input: {golden.input!r}")
                dataset.add_test_case(LLMTestCase(
                    input=golden.input,
                    actual_output=rec.get("actual_output"),
                    expected_output=golden.expected_output,
                    context=rec.get("context", golden.context),
                    retrieval_context=rec.get("retrieval_context", golden.retrieval_context),
                ))
            return dataset.test_cases
        
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
        print_results: bool = True,
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
                show_indicator=show_indicator, print_results=print_results
            ),
            cache_config=CacheConfig(use_cache=use_cache),
            identifier=identifier,
        )

    # ---- convenience ----------------------------------------------------

    @staticmethod
    def available_use_cases() -> List[str]:
        return list_use_cases()
