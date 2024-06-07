from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from benchmark.results import TestCaseResult
from constants import DEFAULT_OUTPUT_PATH
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

    def store_multiple_test_case_results(self, test_case_results: List[TestCaseResult]):
        for test_case_result in test_case_results:
            self.store_test_case_result(test_case_result)

    @staticmethod
    def get_classifications_file_name(model_name: str, cot_uuid: str, non_cot_uuid: str):
        return StorageBackend.get_run_dependant_file_name(model_name, cot_uuid, non_cot_uuid, "classifications", "csv")

    @staticmethod
    def get_samples_file_name(model_name: str, cot_uuid: str, non_cot_uuid: str):
        return StorageBackend.get_run_dependant_file_name(model_name, cot_uuid, non_cot_uuid, "samples", "csv")

    @staticmethod
    def get_run_dependant_file_name(model_name: str, cot_uuid: str, non_cot_uuid: str, suffix: str, extension: str):
        file_name = f"{model_name}" \
                   f"_C-{cot_uuid[:8]}" \
                   f"_N-{non_cot_uuid[:8]}" \
                   f"_{suffix}.{extension}"

        return file_name


STORAGE_BACKEND_REGISTRY = Registry(
    registry_name="storage",
    base_class=StorageBackend,
    lazy_load_dirs=["storage/backends"]
)
