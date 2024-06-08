import json
import os
from pathlib import Path
from typing import List, Dict, Union

from benchmark.results import TestCaseResult, SingleBenchmarkResult, Metrics


class ResultLoader:
    @staticmethod
    def load_result(result_file_path: Union[Path, str]) -> TestCaseResult:
        if type(result_file_path) == str:
            result_file_path = Path(result_file_path)

        if result_file_path.is_dir():
            raise ValueError("Result file path is a directory.")
        elif result_file_path is None:
            raise ValueError("The result file path is invalid.")

        file = open(result_file_path)
        data = json.load(file)

        return ResultLoader.convert_to_test_case_result(data)

    @staticmethod
    def convert_to_test_case_result(data: Dict) -> TestCaseResult:
        if "label_permutation" not in data.keys():
            data["label_permutation"] = ""
        if "n_random" not in data.keys():
            data["n_random"] = -1
        if "random_seed" not in data.keys():
            data["random_seed"] = -1

        return TestCaseResult(
            uuid=data['uuid'],
            model=data['model'],
            dataset_name=data['dataset_name'],
            benchmark_name=data['benchmark_name'],
            label_numbering=data['label_numbering'],
            label_permutation=data['label_permutation'],
            n_random=data['n_random'],
            random_seed=data['random_seed'],
            hostname=data['hostname'],
            inference_backend=data['inference_backend'],
            inference_backend_properties=data.get('inference_backend_properties'),
            metrics=Metrics(**data['metrics']),
            start_time=data['start_time'],
            end_time=data['end_time'],
            execution_seconds=data['execution_seconds'],
            current_commit_hash=data['current_commit_hash'],
            comment=data['comment'],
            use_chat_template=data['use_chat_template'],
            results=[SingleBenchmarkResult(**result) for result in data['results']]
        )

    @staticmethod
    def load_results(result_files_path: Union[Path, str]) -> List[TestCaseResult]:
        if type(result_files_path) == str:
            result_files_path = Path(result_files_path)

        result_files = []

        if result_files_path.is_dir():
            for file in sorted(os.listdir(result_files_path)):
                if file.endswith(".json"):
                    result_files.append(os.path.join(result_files_path, file))
        else:
            result_files.append(str(result_files_path))

        test_case_results: List[TestCaseResult] = []

        for result_file in result_files:
            result_file_path = Path(result_file)
            test_case_results.append(ResultLoader.load_result(result_file_path))

        return test_case_results

    @staticmethod
    def load_two_runs(
            first_result_files_path: Union[Path, str],
            second_result_files_path: Union[Path, str]
    ) -> (List[Dict], List[Dict]):
        first_test_case_results = ResultLoader.load_results(first_result_files_path)
        second_test_case_results = ResultLoader.load_results(second_result_files_path)

        if len(first_test_case_results) != len(second_test_case_results):
            raise ValueError("Number of files in the two dirs are not equal.")

        first_models = [result["model"] for result in first_test_case_results]
        second_models = [result["model"] for result in second_test_case_results]

        if sorted(first_models) != sorted(second_models):
            raise ValueError("No matching result present for each model.")

        return first_test_case_results, second_test_case_results

    @staticmethod
    def all_results_contain_reasoning(results: List[TestCaseResult]) -> bool:
        return all("reasoning" in result["results"][0]["completions"][0] for result in results)

    @staticmethod
    def ensure_reasoning_present(results: List[Dict]):
        if not ResultLoader.all_results_contain_reasoning(results):
            raise ValueError("The CoT path contains result files that do not contain reasoning traces.")
