from typing import Dict, Any, List

from benchmark.results import TestCaseResult
from trace_analysis.classification.trace_class import TraceClass
from trace_analysis.classification.trace_classifier import TraceClassifier
from utils.list_utils import float_list_to_percent
from utils.logger import Logger
from utils.test_case_result_helper import TestCaseResultHelper


class ClassAccuracy:
    @staticmethod
    def compute_all(
            cot_test_case_results: List[TestCaseResult],
            non_cot_test_case_results: List[TestCaseResult],
            class_id: int = -1
    ):
        class_ids = [class_id]
        if class_id == -1:
            class_ids.extend(TraceClass.get_ids())

        for class_id in class_ids:
            stat_table_rows = []

            for result_id in range(len(cot_test_case_results)):
                question_ids_of_class = TraceClassifier.get_question_ids_of_class(
                    class_id=class_id,
                    cot_test_case_result=cot_test_case_results[result_id],
                    non_cot_test_case_result=non_cot_test_case_results[result_id]
                )

                complete_result = ClassAccuracy.compute(
                    cot_test_case_result=cot_test_case_results[result_id],
                    non_cot_test_case_result=non_cot_test_case_results[result_id],
                    question_ids_of_class=question_ids_of_class
                )

                model = complete_result["model"]
                non_cot_accuracy_in_class = complete_result["correct_non_cot_results"] / complete_result["total_results_in_class"]
                cot_accuracy_in_class = complete_result["correct_cot_results"] / complete_result["total_results_in_class"]
                ratio = complete_result["total_results_in_class"] / complete_result["total_results"]

                stat_table_rows.append([model, non_cot_accuracy_in_class, cot_accuracy_in_class, ratio])

            Logger.print_header(f"Trace class accuracy stats, trace class: {'all' if class_id == -1 else class_id}")
            Logger.print_table(
                rows=[float_list_to_percent(stat_table_row) for stat_table_row in stat_table_rows],
                headers=["Model", "Non CoT Accuracy", "CoT Accuracy", "Ratio"]
            )

    @staticmethod
    def compute(
            cot_test_case_result,
            non_cot_test_case_result,
            question_ids_of_class: List[int]
    ) -> Dict[str, Any]:
        correct_answers = TestCaseResultHelper.get_correct_answers(cot_test_case_result)

        cot_model_choices = TestCaseResultHelper.get_model_choices(cot_test_case_result)
        non_cot_model_choices = TestCaseResultHelper.get_model_choices(non_cot_test_case_result)

        if len(cot_model_choices) != len(non_cot_model_choices):
            raise ValueError("Lengths don't match")

        correct_cot_ids = []
        correct_non_cot_ids = []

        for index in range(len(cot_model_choices)):
            if index not in question_ids_of_class:
                continue

            if cot_model_choices[index] == correct_answers[index]:
                correct_cot_ids.append(index)

            if non_cot_model_choices[index] == correct_answers[index]:
                correct_non_cot_ids.append(index)

        runs_match_result = {
            "model": cot_test_case_result["model"],
            "total_results": len(cot_model_choices),
            "total_results_in_class": len(question_ids_of_class),
            "correct_cot_results": len(correct_cot_ids),
            "correct_non_cot_results": len(correct_non_cot_ids)
        }

        return runs_match_result
