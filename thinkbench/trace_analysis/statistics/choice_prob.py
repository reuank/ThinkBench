import statistics
from typing import List

import numpy as np

from benchmark.results import TestCaseResult
from trace_analysis.classification.trace_classifier import TraceClassifier
from utils.list_utils import only_keep_indexes, remove_indexes, list_intersection
from utils.logger import Logger
from utils.test_case_result_helper import TestCaseResultHelper


class ChoiceProb:
    @staticmethod
    def compute_all(
            cot_test_case_results: List[TestCaseResult],
            non_cot_test_case_results: List[TestCaseResult],
            class_id: int = -1
    ):
        cot_table_rows = []
        non_cot_table_rows = []

        part_to_analyze = ["all_in_class", "correct_in_class", "incorrect_in_class"][2]

        for test_case_result_id, cot_test_case_result in enumerate(cot_test_case_results):
            cot_model_choice_probs, non_cot_model_choice_probs = ChoiceProb.compute(
                cot_test_case_result=cot_test_case_result,
                non_cot_test_case_result=non_cot_test_case_results[test_case_result_id],
                class_id=class_id,
                part_to_analyze=part_to_analyze
            )

            cot_table_rows.append(
                ChoiceProb.get_model_choice_prob_table_row(
                    model_name=cot_test_case_result["model"],
                    model_choice_probs=cot_model_choice_probs
                )
            )

            non_cot_table_rows.append(
                ChoiceProb.get_model_choice_prob_table_row(
                    model_name=cot_test_case_result["model"],
                    model_choice_probs=non_cot_model_choice_probs
                )
            )

        Logger.print_header(f"{cot_test_case_results[0]['benchmark_name']} Model Choice Probs (Dataset: {cot_test_case_results[0]['dataset_name']}, "
                            f"Class ID: {'all' if class_id == -1 else class_id}, "
                            f"Subsection: {part_to_analyze.replace('_', ' ')})")
        Logger.print_table(rows=cot_table_rows, headers=ChoiceProb.get_model_choice_prob_table_header())

        Logger.print_header(f"{non_cot_test_case_results[0]['benchmark_name']} Model Choice Probs (Dataset: {non_cot_test_case_results[0]['dataset_name']}, "
                            f"Class ID: {'all' if class_id == -1 else class_id}, "
                            f"Subsection: {part_to_analyze.replace('_', ' ')})")
        Logger.print_table(rows=non_cot_table_rows, headers=ChoiceProb.get_model_choice_prob_table_header())

    @staticmethod
    def compute(
            cot_test_case_result: TestCaseResult,
            non_cot_test_case_result: TestCaseResult,
            class_id: int = -1,
            part_to_analyze: str = "all_in_class"
    ) -> (List[float], List[float]):
        question_ids_of_class = TraceClassifier.get_question_ids_of_class(
            class_id=class_id,
            cot_test_case_result=cot_test_case_result,
            non_cot_test_case_result=non_cot_test_case_result
        )

        if part_to_analyze == "all_in_class":
            cot_indexes_to_keep = question_ids_of_class
            non_cot_indexes_to_keep = question_ids_of_class
        elif part_to_analyze == "correct_in_class":
            cot_indexes_to_keep = TestCaseResultHelper.get_question_ids_of_correct_model_choices(
                test_case_result=cot_test_case_result
            )

            non_cot_indexes_to_keep = TestCaseResultHelper.get_question_ids_of_correct_model_choices(
                test_case_result=non_cot_test_case_result
            )
        elif part_to_analyze == "incorrect_in_class":
            cot_indexes_to_keep = TestCaseResultHelper.get_question_ids_of_incorrect_model_choices(
                test_case_result=cot_test_case_result
            )

            non_cot_indexes_to_keep = TestCaseResultHelper.get_question_ids_of_incorrect_model_choices(
                test_case_result=non_cot_test_case_result
            )
        else:
            raise ValueError(f"Part to analyze {part_to_analyze} not defined.")

        cot_indexes_to_keep = list_intersection(cot_indexes_to_keep, question_ids_of_class)
        non_cot_indexes_to_keep = list_intersection(non_cot_indexes_to_keep, question_ids_of_class)

        cot_model_choice_probs = only_keep_indexes(
            from_list=ChoiceProb.get_model_choice_probs(cot_test_case_result),
            indexes_to_keep=cot_indexes_to_keep
        )

        non_cot_model_choice_probs = only_keep_indexes(
            from_list=ChoiceProb.get_model_choice_probs(non_cot_test_case_result),
            indexes_to_keep=non_cot_indexes_to_keep
        )

        return cot_model_choice_probs, non_cot_model_choice_probs

    @staticmethod
    def get_model_choice_probs(test_case_result: TestCaseResult):
        model_choices = TestCaseResultHelper.get_model_choices(test_case_result)
        model_completion_probs = TestCaseResultHelper.get_model_choice_logprobs(test_case_result)
        assert len(model_choices) == len(model_completion_probs)

        model_choice_probs = []

        for index in range(len(model_choices)):
            # Try some variations of the label token
            model_choice_prob = model_completion_probs[index].get(model_choices[index], 0.0)
            if model_choice_prob == 0.0:
                model_choice_prob = model_completion_probs[index].get(" "+model_choices[index], 0.0)
            if model_choice_prob == 0.0:
                model_choice_prob = model_completion_probs[index].get(model_choices[index]+" ", 0.0)
            # if model_choice_prob == 0.0:
            #     Logger.error(f"Could not extract label prob for question id {index}, "
            #                  f"choice {model_choices[index]} and model {cot_test_case_result['model']}.")

            model_choice_probs.append(model_choice_prob)

        return model_choice_probs

    @staticmethod
    def get_model_choice_prob_table_header():
        return ["Model", "Average Label Prob", "Min Label Prob", "Max Label Prob", "10th Percentile", "90th Percentile"]

    @staticmethod
    def get_model_choice_prob_table_row(model_name: str, model_choice_probs: List[float]):
        return [
            model_name,
            f"{statistics.fmean(model_choice_probs):.4%} ({len(model_choice_probs)})",
            f"{min(model_choice_probs):.4%}",  # (ID {model_choice_probs.index(min(model_choice_probs))})",
            f"{max(model_choice_probs):.4%}",  # (ID {model_choice_probs.index(max(model_choice_probs))})",
            f"{np.percentile(model_choice_probs, 10):.4%}",
            f"{np.percentile(model_choice_probs, 90):.4%}",
        ]
