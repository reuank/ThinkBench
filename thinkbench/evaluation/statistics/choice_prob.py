import statistics
from typing import List

import numpy as np

from benchmark.results import TestCaseResult
from evaluation.statistics.run_stat import RunStat
from utils.list_utils import only_keep_indexes
from utils.logger import Logger
from utils.test_case_result_helper import TestCaseResultHelper


class ChoiceProb(RunStat):
    @staticmethod
    def compute_all(
            cot_test_case_results: List[TestCaseResult],
            non_cot_test_case_results: List[TestCaseResult],
            class_id: int = -1,
            **kwargs
    ):
        cot_table_rows = []
        non_cot_table_rows = []

        class_part: str = kwargs.get("class_part", "all_in_class")

        for test_case_result_id, cot_test_case_result in enumerate(cot_test_case_results):
            cot_model_choice_probs, non_cot_model_choice_probs = ChoiceProb.compute(
                cot_test_case_result=cot_test_case_result,
                non_cot_test_case_result=non_cot_test_case_results[test_case_result_id],
                class_id=class_id,
                class_part=class_part
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
                            f"Subsection: {class_part.replace('_', ' ')})")
        Logger.print_table(rows=cot_table_rows, headers=ChoiceProb.get_model_choice_prob_table_header())

        Logger.print_header(f"{non_cot_test_case_results[0]['benchmark_name']} Model Choice Probs (Dataset: {non_cot_test_case_results[0]['dataset_name']}, "
                            f"Class ID: {'all' if class_id == -1 else class_id}, "
                            f"Subsection: {class_part.replace('_', ' ')})")
        Logger.print_table(rows=non_cot_table_rows, headers=ChoiceProb.get_model_choice_prob_table_header())

    @staticmethod
    def compute(
            cot_test_case_result: TestCaseResult,
            non_cot_test_case_result: TestCaseResult,
            class_id: int = -1,
            class_part: str = "all_in_class"
    ) -> (List[float], List[float]):
        cot_model_choice_probs = TestCaseResultHelper.get_model_choice_probs(cot_test_case_result)
        non_cot_model_choice_probs = TestCaseResultHelper.get_model_choice_probs(non_cot_test_case_result)

        cot_indexes_to_keep, non_cot_indexes_to_keep = RunStat.get_class_part_question_ids(
            cot_test_case_result=cot_test_case_result,
            non_cot_test_case_result=non_cot_test_case_result,
            class_id=class_id,
            class_part=class_part
        )

        cot_model_choice_probs = only_keep_indexes(
            from_list=cot_model_choice_probs,
            indexes_to_keep=cot_indexes_to_keep
        )

        non_cot_model_choice_probs = only_keep_indexes(
            from_list=non_cot_model_choice_probs,
            indexes_to_keep=non_cot_indexes_to_keep
        )

        return cot_model_choice_probs, non_cot_model_choice_probs

    @staticmethod
    def get_model_choice_prob_table_header():
        return ["Model", "Average Choice Prob", "Min Choice Prob", "Max Choice Prob", "10th Percentile", "90th Percentile"]

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
