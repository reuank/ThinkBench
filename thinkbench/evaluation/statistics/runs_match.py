from typing import List, Dict, Any

import numpy as np

from benchmark.results import TestCaseResult
from evaluation.statistics.run_stat import RunStat
from utils.list_utils import float_list_to_percent
from utils.logger import Logger
from utils.test_case_result_helper import TestCaseResultHelper


class RunComparison(RunStat):
    @staticmethod
    def compute_all(
            cot_test_case_results: List[TestCaseResult],
            non_cot_test_case_results: List[TestCaseResult],
            class_id: int = -1,
            **kwargs
    ):
        for result_id in range(len(cot_test_case_results)):
            cot_question_ids_of_class, non_cot_question_ids_of_class = RunStat.get_class_part_question_ids(
                cot_test_case_result=cot_test_case_results[result_id],
                non_cot_test_case_result=non_cot_test_case_results[result_id],
                class_id=class_id,
                class_part="all_in_class"
            )

            complete_result = RunComparison.compute(
                cot_test_case_result=cot_test_case_results[result_id],
                non_cot_test_case_result=non_cot_test_case_results[result_id],
                question_ids_of_class=cot_question_ids_of_class
            )

            cot_epistemic_cost = complete_result["mismatches_non_cot_superior"]
            cot_epistemic_gain = complete_result["mismatches_cot_superior"]
            both_wrong = complete_result["matches_incorrect"] + complete_result["mismatches_none_superior"]
            both_correct = complete_result["matches_correct"]

            absolute_rows = np.array([
                [both_correct, cot_epistemic_cost],
                [cot_epistemic_gain, both_wrong]
            ])

            # replace complete_result["results"] with len(question_ids_of_class) for numbers relative to number of q's in class
            relative_rows = (absolute_rows / complete_result["num_results"]).tolist()

            row_headers = ["Baseline Correct", "Baseline Incorrect"]
            column_headers = ["CoT Correct", "CoT Incorrect"]

            Logger.print_header(f"Runs Match Stats for model {complete_result['model']}, trace class: {'all' if class_id == -1 else class_id}")
            Logger.print_table(
                rows=[[row_headers[i]] + float_list_to_percent(row) for i, row in enumerate(relative_rows)],
                headers=[""] + column_headers
            )

    @staticmethod
    def compute(
            cot_test_case_result: TestCaseResult,
            non_cot_test_case_result: TestCaseResult,
            question_ids_of_class: List[int]
    ) -> Dict[str, Any]:
        correct_answers = TestCaseResultHelper.get_correct_answers(cot_test_case_result)

        cot_model_choices = TestCaseResultHelper.get_model_choices(cot_test_case_result)
        non_cot_model_choices = TestCaseResultHelper.get_model_choices(non_cot_test_case_result)

        if len(cot_model_choices) != len(non_cot_model_choices):
            raise ValueError("Lengths don't match")

        matches_ids = []
        matches_correct_ids = []
        matches_incorrect_ids = []

        mismatch_ids = []
        mismatch_cot_superior_ids = []
        mismatch_non_cot_superior_ids = []
        mismatch_none_superior_ids = []

        for index in range(len(cot_model_choices)):
            if index not in question_ids_of_class:
                continue

            # Disagreement
            if cot_model_choices[index] != non_cot_model_choices[index]:
                mismatch_ids.append(index)
                if cot_model_choices[index] == correct_answers[index]:
                    mismatch_cot_superior_ids.append(index)
                elif non_cot_model_choices[index] == correct_answers[index]:
                    mismatch_non_cot_superior_ids.append(index)
                else:
                    mismatch_none_superior_ids.append(index)

            # Agreement
            else:
                matches_ids.append(index)
                if cot_model_choices[index] == correct_answers[index]:
                    matches_correct_ids.append(index)
                else:
                    matches_incorrect_ids.append(index)

        runs_match_result = {
            "model": cot_test_case_result["model"],
            "num_results": len(cot_model_choices),
            "non_cot_uuid": non_cot_test_case_result["uuid"],
            "cot_uuid": cot_test_case_result["uuid"],
            "non_cot_accuracy": non_cot_test_case_result["metrics"]["accuracy"],
            "cot_accuracy": cot_test_case_result["metrics"]["accuracy"],
            "matches": len(matches_ids),
            "matches_correct": len(matches_correct_ids),
            "matches_incorrect": len(matches_incorrect_ids),
            "mismatches": len(mismatch_ids),
            "mismatches_cot_superior": len(mismatch_cot_superior_ids),
            "mismatches_non_cot_superior": len(mismatch_non_cot_superior_ids),
            "mismatches_none_superior": len(mismatch_none_superior_ids),
        }

        return runs_match_result
