import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from benchmark.results import TestCaseResult
from utils.registry import Registry


class StorageBackend(ABC):
    def __init__(self):
        try:
            self.output_path = os.environ.get("TB_OUTPUT_PATH")
            if not self.output_path:
                raise KeyError
            else:
                self.output_path = Path(self.output_path)
                self.results_path = self.output_path / "test_runs"
                self.analysis_path = self.output_path / "analysis"

                for folder in [self.results_path, self.analysis_path]:
                    folder.mkdir(parents=True, exist_ok=True)
        except KeyError:
            print("Please specify an output path. Did you forget to source .env?")
            exit()

    @abstractmethod
    def store_test_case_result(self, test_case_result: TestCaseResult):
        raise NotImplementedError

    def store_multiple_test_case_results(self, test_case_results: List[TestCaseResult]):
        for test_case_result in test_case_results:
            self.store_test_case_result(test_case_result)

    @staticmethod
    def get_samples_filename(model_name: str, cot_uuid: str, non_cot_uuid: str):
        filename = f"{model_name}" \
                   f"_C-{cot_uuid[:8]}" \
                   f"_N-{non_cot_uuid[:8]}" \
                   f"_samples.csv"

        return filename

    @staticmethod
    def get_classifications_filename(model_name: str, cot_uuid: str, non_cot_uuid: str):
        filename = f"{model_name}" \
                   f"_C-{cot_uuid[:8]}" \
                   f"_N-{non_cot_uuid[:8]}" \
                   f"_classification.csv"

        return filename


STORAGE_BACKEND_REGISTRY = Registry(
    registry_name="storage",
    base_class=StorageBackend,
    lazy_load_dirs=["storage/backends"]
)
