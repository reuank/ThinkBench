import statistics
from typing import List

from benchmark.results import TestCaseResult
from trace_analysis.statistics.run_stat import RunStat
from utils.list_utils import only_keep_indexes
from utils.logger import Logger
from utils.test_case_result_helper import TestCaseResultHelper


class LabelProbs(RunStat):
    @staticmethod
    def compute_all(
            cot_test_case_results: List[TestCaseResult],
            non_cot_test_case_results: List[TestCaseResult],
            class_id: int = -1,
            **kwargs
    ):
        class_part: str = kwargs.get("class_part", "all_in_class")

        table_rows = []

        for test_case_result_id, cot_test_case_result in enumerate(cot_test_case_results):
            cot_label_probs_list, non_cot_label_probs_list = LabelProbs.compute(
                cot_test_case_result=cot_test_case_result,
                non_cot_test_case_result=non_cot_test_case_results[test_case_result_id],
                class_id=class_id,
                class_part=class_part
            )

            cot_label_probs_sums: List[float] = []
            for cot_label_probs in cot_label_probs_list:
                cot_label_probs_sums.append(sum(cot_label_probs.values()))

            non_cot_label_probs_sums: List[float] = []
            for non_cot_label_probs in non_cot_label_probs_list:
                non_cot_label_probs_sums.append(sum(non_cot_label_probs.values()))

            table_rows.append([
                cot_test_case_result["model"],
                f"{statistics.fmean(cot_label_probs_sums):.2%}",
                f"{statistics.fmean(non_cot_label_probs_sums):.2%}"
            ])

        Logger.print_table(rows=table_rows, headers=["Model", "CoT Avg Label Prob Mass", "Non-CoT Avg Label Prob Mass"])


    @staticmethod
    def compute(
            cot_test_case_result: TestCaseResult,
            non_cot_test_case_result: TestCaseResult,
            class_id: int = -1,
            class_part: str = "all_in_class"
    ) -> (List[float], List[float]):
        cot_label_probs = TestCaseResultHelper.get_label_probs(cot_test_case_result)
        non_cot_label_probs = TestCaseResultHelper.get_label_probs(non_cot_test_case_result)

        cot_indexes_to_keep, non_cot_indexes_to_keep = RunStat.get_class_part_question_ids(
            cot_test_case_result=cot_test_case_result,
            non_cot_test_case_result=non_cot_test_case_result,
            class_id=class_id,
            class_part=class_part
        )

        only_keep_indexes(
            from_list=cot_label_probs,
            indexes_to_keep=cot_indexes_to_keep
        )

        non_cot_label_probs = only_keep_indexes(
            from_list=non_cot_label_probs,
            indexes_to_keep=non_cot_indexes_to_keep
        )

        return cot_label_probs, non_cot_label_probs
