from abc import ABC, abstractmethod
from typing import List, Dict, Any

from benchmark.results import TestCaseResult
from trace_analysis.classification.trace_classifier import TraceClassifier
from utils.list_utils import list_intersection
from utils.test_case_result_helper import TestCaseResultHelper


class RunStat(ABC):
    @staticmethod
    @abstractmethod
    def compute_all(
        cot_test_case_result: TestCaseResult,
        non_cot_test_case_result: TestCaseResult,
        class_id: int = -1,
        **kwargs
    ) -> Any:
        raise NotImplementedError

    @staticmethod
    def get_class_part_question_ids(
        cot_test_case_result: TestCaseResult,
        non_cot_test_case_result: TestCaseResult,
        class_id: int = -1,
        class_part: str = "all_in_class"
    ) -> (List[int], List[int]):
        question_ids_of_class = TraceClassifier.get_question_ids_of_class(
            class_id=class_id,
            cot_test_case_result=cot_test_case_result,
            non_cot_test_case_result=non_cot_test_case_result
        )

        if class_part == "all_in_class":
            cot_indexes_to_keep = question_ids_of_class
            non_cot_indexes_to_keep = question_ids_of_class
        elif class_part == "correct_in_class":
            cot_indexes_to_keep = TestCaseResultHelper.get_question_ids_of_correct_model_choices(
                test_case_result=cot_test_case_result
            )

            non_cot_indexes_to_keep = TestCaseResultHelper.get_question_ids_of_correct_model_choices(
                test_case_result=non_cot_test_case_result
            )
        elif class_part == "incorrect_in_class":
            cot_indexes_to_keep = TestCaseResultHelper.get_question_ids_of_incorrect_model_choices(
                test_case_result=cot_test_case_result
            )

            non_cot_indexes_to_keep = TestCaseResultHelper.get_question_ids_of_incorrect_model_choices(
                test_case_result=non_cot_test_case_result
            )
        else:
            raise ValueError(f"Part to analyze {class_part} not defined.")

        cot_indexes_to_keep = list_intersection(cot_indexes_to_keep, question_ids_of_class)
        non_cot_indexes_to_keep = list_intersection(non_cot_indexes_to_keep, question_ids_of_class)

        return cot_indexes_to_keep, non_cot_indexes_to_keep
