import json
import os
from pathlib import Path


class ResultLoader:
    @staticmethod
    def load_result_file(result_file: str):
        result_file_path: Path = Path(result_file)

        if result_file_path.is_dir():
            raise ValueError("Result file path is a directory.")
        elif result_file_path is None:
            raise ValueError("The result file path is invalid.")

        file = open(result_file_path)
        data = json.load(file)

        return data

    @staticmethod
    def load_result_files_from_dir(results_dir: str):
        result_files = []

        for file in sorted(os.listdir(results_dir)):
            if file.endswith(".json"):
                result_files.append(os.path.join(results_dir, file))

        models = [result_file.split("/")[-1].split("_")[3] for result_file in result_files]

        result_files_data = []
        for result_file in result_files:
            result_files_data.append(ResultLoader.load_result_file(result_file))

        return models, result_files_data

    @staticmethod
    def load_cot_and_non_cot_from_dirs(cot_results_dir: str, non_cot_results_dir: str):
        cot_models, cot_results_data = ResultLoader.load_result_files_from_dir(cot_results_dir)
        non_cot_models, non_cot_results_data = ResultLoader.load_result_files_from_dir(non_cot_results_dir)

        if len(cot_results_data) != len(non_cot_results_data):
            raise ValueError("Number of CoT and Non-CoT files are not equal.")

        if sorted(cot_models) != sorted(non_cot_models):
            raise ValueError("No CoT and Non-CoT result present for each model.")

        return cot_models, non_cot_results_data, cot_results_data

