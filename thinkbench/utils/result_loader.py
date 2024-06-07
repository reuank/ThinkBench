import json
import os
from pathlib import Path
from typing import List, Dict, Union


class ResultLoader:
    @staticmethod
    def load_result(result_file_path: Union[Path, str]) -> Dict:
        if type(result_file_path) == str:
            result_file_path = Path(result_file_path)

        if result_file_path.is_dir():
            raise ValueError("Result file path is a directory.")
        elif result_file_path is None:
            raise ValueError("The result file path is invalid.")

        file = open(result_file_path)
        data = json.load(file)

        return data

    @staticmethod
    def load_results(result_files_path: Union[Path, str]) -> List[Dict]:
        if type(result_files_path) == str:
            result_files_path = Path(result_files_path)

        result_files = []

        if result_files_path.is_dir():
            for file in sorted(os.listdir(result_files_path)):
                if file.endswith(".json"):
                    result_files.append(os.path.join(result_files_path, file))
        else:
            result_files.append(str(result_files_path))

        result_files_data = []
        for result_file in result_files:
            result_file_path = Path(result_file)
            result_files_data.append(ResultLoader.load_result(result_file_path))

        return result_files_data

    @staticmethod
    def load_two_runs(
            first_result_files_path: Union[Path, str],
            second_result_files_path: Union[Path, str]
    ) -> (List[Dict], List[Dict]):
        first_results_data = ResultLoader.load_results(first_result_files_path)
        second_results_data = ResultLoader.load_results(second_result_files_path)

        if len(first_results_data) != len(second_results_data):
            raise ValueError("Number of files in the two dirs are not equal.")

        first_models = [result["model"] for result in first_results_data]
        second_models = [result["model"] for result in second_results_data]

        if sorted(first_models) != sorted(second_models):
            raise ValueError("No matching result present for each model.")

        return first_results_data, second_results_data

    @staticmethod
    def all_results_contain_reasoning(results: List[Dict]) -> bool:
        return all("reasoning" in result["results"][0]["completions"][0] for result in results)

    @staticmethod
    def ensure_reasoning_present(results: List[Dict]):
        if not ResultLoader.all_results_contain_reasoning(results):
            raise ValueError("The CoT path contains result files that do not contain reasoning traces.")
