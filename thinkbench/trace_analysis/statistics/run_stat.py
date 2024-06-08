from abc import ABC, abstractmethod
from typing import Dict, List

from benchmark.results import TestCaseResult
from storage.backends.csv_file_storage import CsvFileStorage
from storage.backends.json_file_storage import JsonFileStorage


class RunStat(ABC):
    @staticmethod
    @abstractmethod
    def compute_all(
            cot_test_case_results: List[TestCaseResult],
            non_cot_test_case_results: List[TestCaseResult],
            class_id: int = -1
    ):
        raise NotImplementedError

    @staticmethod
    def get_model_choices(result_file_data: Dict) -> List[str]:
        return [single_result["model_choice"] for single_result in result_file_data["results"]]

    @staticmethod
    def get_correct_answers(result_file_data: Dict) -> List[str]:
        return [single_result["correct_answer"] for single_result in result_file_data["results"]]

    @staticmethod
    def float_list_to_percent(float_list: List[float]) -> List[str]:
        return [f"{value:.2%}" if isinstance(value, float) else value for value in float_list]

    @staticmethod
    def get_question_ids_of_class(
            class_id: int,
            cot_test_case_result: TestCaseResult,
            non_cot_test_case_result: TestCaseResult
    ):
        json_file_storage = JsonFileStorage()

        classification_result_file_name = json_file_storage.get_classification_result_file_name_for_test_cases(
            non_cot_test_case_result=non_cot_test_case_result,
            cot_test_case_result=cot_test_case_result
        )

        classification_result = json_file_storage.load_classification_result(classification_result_file_name)

        question_ids_of_class = []
        for single_classification in classification_result["results"]:
            if class_id == -1:
                question_ids_of_class.append(single_classification["question_id"])
            else:
                if class_id == single_classification["automatic_class_id"]:
                    question_ids_of_class.append(single_classification["question_id"])

        return question_ids_of_class
