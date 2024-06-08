import random
from abc import ABC, abstractmethod
from typing import Dict, List, Union

from benchmark.results import TestCaseResult
from constants import TRACE_SAMPLES_PER_RUN
from storage.backends.json_file_storage import JsonFileStorage
from trace_analysis.classification.automatic_trace_classifier import TraceClass
from trace_analysis.classification.classification_result import ClassificationResult, SingleClassification
from utils.logger import Logger


class TraceClassifier(ABC):
    @staticmethod
    @abstractmethod
    def classify_traces(
            all_cot_results: List[TestCaseResult],
            all_non_cot_results: List[TestCaseResult],
            override: bool
    ) -> List[ClassificationResult]:
        raise NotImplementedError

    @staticmethod
    def is_trace_class(value) -> bool:
        try:
            value = int(value)
            for trace_class in TraceClass:
                if trace_class.value == value:
                    return True
            return False
        except ValueError:
            return False

    @staticmethod
    def get_model_choices(test_case_result: TestCaseResult) -> List[str]:
        return [single_benchmark_result["model_choice"] for single_benchmark_result in test_case_result["results"]]

    @staticmethod
    def get_manual_class_ids(classification_result: ClassificationResult):
        return [
                    0 if single_classification["manual_class_id"] is None
                    else single_classification["manual_class_id"]
                    for single_classification in classification_result["results"]
                ]

    @staticmethod
    def get_automatic_class_ids(classification_result: ClassificationResult):
        return [
            0 if single_classification["automatic_class_id"] is None
            else single_classification["automatic_class_id"]
            for single_classification in classification_result["results"]
        ]

    @staticmethod
    def replace_results_with_samples(test_case_result: TestCaseResult, n_samples: int = TRACE_SAMPLES_PER_RUN, seed: str = "") -> Dict:
        single_benchmark_results = test_case_result["results"]
        random.seed(seed)
        single_benchmark_result_samples = random.sample(single_benchmark_results, n_samples)

        test_case_result["results"] = single_benchmark_result_samples

        return test_case_result

    @staticmethod
    def get_single_classification_for_question_id(
            question_id: int,
            classification_result: ClassificationResult
    ) -> Union[SingleClassification, None]:
        for single_classification in classification_result["results"]:
            if single_classification.get("question_id") == question_id:
                return single_classification

        return None

    @staticmethod
    def merge_manual_class_ids(
            manual_classification_result: Union[ClassificationResult, None],
            automatic_classification_result: ClassificationResult
    ) -> ClassificationResult:
        if not manual_classification_result:
            return automatic_classification_result

        for automatic_single_classification in automatic_classification_result["results"]:
            question_id = automatic_single_classification["question_id"]
            matching_manual_single_classification = TraceClassifier.get_single_classification_for_question_id(
                question_id=question_id,
                classification_result=manual_classification_result
            )

            if matching_manual_single_classification:
                automatic_single_classification["manual_class_id"] = matching_manual_single_classification["manual_class_id"]

        return automatic_classification_result

    @staticmethod
    def store_classification_results(classification_results: List[ClassificationResult]):
        json_file_storage_backend: JsonFileStorage = JsonFileStorage()
        written_files_count = 0

        for classification_result in classification_results:
            json_file_storage_backend.store_classification_result(classification_result, verbose=False)
            written_files_count += 1

        if written_files_count > 0:
            Logger.info(f"Classification files were written to {json_file_storage_backend.analysis_path}")
