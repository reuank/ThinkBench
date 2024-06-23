from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from benchmark.results import TestCaseResult
from constants import DEFAULT_OUTPUT_PATH
from evaluation.classification.classification_result import ClassificationResult
from utils.env_loader import EnvReader
from utils.registry import Registry


class StorageBackend(ABC):
    def __init__(self):
        try:
            self.output_path = EnvReader.get("TB_OUTPUT_PATH", DEFAULT_OUTPUT_PATH)
            if not self.output_path:
                raise KeyError
            else:
                self.output_path = Path(self.output_path)
                self.results_path = self.output_path / "test_runs"
                self.analysis_path = self.output_path / "analysis"

                for folder in [self.results_path, self.analysis_path]:
                    folder.mkdir(parents=True, exist_ok=True)
        except KeyError:
            print("Please specify an output path, either in constants.py or in the .env file .")
            exit()

    @abstractmethod
    def store_test_case_result(self, test_case_result: TestCaseResult):
        raise NotImplementedError

    @abstractmethod
    def store_classification_result(self, classification_result: ClassificationResult):
        raise NotImplementedError

    def store_multiple_test_case_results(self, test_case_results: List[TestCaseResult]):
        for test_case_result in test_case_results:
            self.store_test_case_result(test_case_result)

    def store_multiple_classification_results(self, classification_results: List[ClassificationResult]):
        for classification_result in classification_results:
            self.store_classification_result(classification_result)

    @staticmethod
    def get_run_dependent_file_name(
            model_name: str,
            benchmark_name: str,
            dataset_name: str,
            cot_uuid: str,
            non_cot_uuid: str,
            extension: str,
            prefix: str = "",
            suffix: str = ""
    ):
        file_name = f"{prefix}" \
                    f"{model_name}" \
                    f"_{benchmark_name}" \
                    f"_{dataset_name}" \
                    f"_C-{cot_uuid}" \
                    f"_N-{non_cot_uuid}" \
                    f"{suffix}.{extension}"

        return file_name


STORAGE_BACKEND_REGISTRY = Registry(
    registry_name="storage",
    base_class=StorageBackend,
    lazy_load_dirs=["storage/backends"]
)
