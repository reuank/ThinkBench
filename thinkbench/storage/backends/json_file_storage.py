import datetime
import json
from typing import Union, Dict
from pathlib import Path

from storage.storage_backend import StorageBackend, STORAGE_BACKEND_REGISTRY
from benchmark.results import TestCaseResult
from trace_analysis.classification.classification_result import ClassificationResult, SingleClassification
from utils.encoders import TotalResultEncoder
from utils.logger import Logger


@STORAGE_BACKEND_REGISTRY.register(name="json_file_storage", is_default=True)
class JsonFileStorage(StorageBackend):
    @staticmethod
    def write_file(file_name, data: Dict, verbose: bool = True):
        f = open(file_name, "w")
        f.write(json.dumps(data, cls=TotalResultEncoder, indent=2))
        f.close()

        if verbose:
            Logger.info(f"File {file_name} written.")

    def store_test_case_result(self, test_case_result: TestCaseResult):
        file_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" \
                    f"_{test_case_result['benchmark_name']}" \
                    f"_{test_case_result['model']}" \
                    f"_{test_case_result['dataset_name']}-{test_case_result['metrics']['total_results']}" \
                    f"_labels-{test_case_result['label_numbering']}" \
                    f"_{'use-chat-template' if test_case_result['use_chat_template'] else 'no-chat-template'}" \
                    f"_{test_case_result['inference_backend']}" \
                    f"_{test_case_result['hostname']}.json"

        JsonFileStorage.write_file(self.results_path / file_name, test_case_result)

    def store_classification_result(self, classification_result: ClassificationResult, verbose: bool = True):
        file_name = self.get_classification_result_file_name(classification_result)

        JsonFileStorage.write_file(self.analysis_path / file_name, classification_result, verbose)

    @staticmethod
    def get_classification_result_file_name(classification_result: ClassificationResult):
        file_name = f"{classification_result['model']}" \
                    f"_{classification_result['cot_benchmark_name']}" \
                    f"_{classification_result['dataset_name']}" \
                    f"_C-{classification_result['cot_uuid']}" \
                    f"_N-{classification_result['non_cot_uuid']}" \
                    f"_classification.json"

        return file_name

    @staticmethod
    def get_classification_result_file_name_by_params(
            model_name: str,
            benchmark_name: str,
            dataset_name: str,
            cot_uuid: str,
            non_cot_uuid: str,
    ):
        return StorageBackend.get_run_dependent_file_name(
            model_name=model_name,
            benchmark_name=benchmark_name,
            dataset_name=dataset_name,
            cot_uuid=cot_uuid,
            non_cot_uuid=non_cot_uuid,
            prefix="",
            suffix="_classification",
            extension="json"
        )

    @staticmethod
    def get_classification_result_file_name_for_test_cases(
            cot_test_case_result: TestCaseResult,
            non_cot_test_case_result: TestCaseResult
    ):
        return JsonFileStorage.get_classification_result_file_name_by_params(
            benchmark_name=cot_test_case_result["benchmark_name"],
            dataset_name=cot_test_case_result["dataset_name"],
            model_name=cot_test_case_result["model"],
            cot_uuid=cot_test_case_result["uuid"],
            non_cot_uuid=non_cot_test_case_result["uuid"],
        )

    @staticmethod
    def load_raw(path: Union[Path, str]):
        if type(path) == str:
            path = Path(path)
        if path.is_dir():
            raise ValueError("The file path is a directory.")
        elif path is None:
            raise ValueError("The file path is invalid.")

        file = open(path)

        return json.load(file)

    @staticmethod
    def convert_dict_to_classification_result(data: Dict) -> ClassificationResult:
        return ClassificationResult(
            model=data["model"],
            cot_uuid=data["cot_uuid"],
            non_cot_uuid=data["non_cot_uuid"],
            cot_benchmark_name=data["cot_benchmark_name"],
            non_cot_benchmark_name=data["non_cot_benchmark_name"],
            dataset_name=data["dataset_name"],
            results=[SingleClassification(**result) for result in data['results']],
        )

    def classification_result_file_exists(self, classification_result: ClassificationResult):
        return (self.analysis_path / self.get_classification_result_file_name(classification_result)).is_file()

    def load_classification_result(self, classification_result_file_name: str) -> ClassificationResult:
        file = open(self.analysis_path / classification_result_file_name)
        data = json.load(file)

        return JsonFileStorage.convert_dict_to_classification_result(data)
