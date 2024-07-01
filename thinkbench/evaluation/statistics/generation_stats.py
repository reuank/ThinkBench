import datetime
from typing import List

from benchmark.results import TestCaseResult
from evaluation.statistics.run_stat import RunStat
from utils.logger import Logger


class GenerationStats(RunStat):
    @staticmethod
    def compute_all(
            cot_test_case_results: List[TestCaseResult],
            non_cot_test_case_results: List[TestCaseResult],
            class_id: int = -1,
            **kwargs
    ):
        table_rows = []

        total_non_cot_seconds = 0
        total_non_cot_tokens = 0
        total_cot_seconds = 0
        total_cot_tokens = 0

        for test_case_result_id, cot_test_case_result in enumerate(cot_test_case_results):
            non_cot_test_case_result = non_cot_test_case_results[test_case_result_id]

            non_cot_tokens = non_cot_test_case_result['metrics']['total_generated_tokens']
            non_cot_seconds = non_cot_test_case_result['execution_seconds']
            cot_tokens = cot_test_case_result['metrics']['total_generated_tokens']
            cot_seconds = cot_test_case_result['execution_seconds']

            table_rows.append([
                cot_test_case_result["model"],
                f"{non_cot_tokens}",
                f"{datetime.timedelta(seconds=non_cot_seconds)}",
                f"{cot_tokens}",
                f"{datetime.timedelta(seconds=cot_seconds)}"
            ])

            total_non_cot_tokens += non_cot_tokens
            total_non_cot_seconds += non_cot_seconds
            total_cot_tokens += cot_tokens
            total_cot_seconds += cot_seconds

        num_non_cot = len(non_cot_test_case_results)
        num_cot = len(cot_test_case_results)
        num_questions = cot_test_case_results[0]['metrics']['total_results']

        table_rows.append([
            "Average",
            f"{total_non_cot_tokens / num_non_cot}",
            f"{datetime.timedelta(seconds=(total_non_cot_seconds/num_non_cot))}",
            f"{total_cot_tokens / num_cot}",
            f"{datetime.timedelta(seconds=(total_cot_seconds/num_cot))}"
        ])

        table_rows.append([
            "Average Per Question",
            f"{(total_non_cot_tokens / num_non_cot) / num_questions}",
            f"{datetime.timedelta(seconds=((total_non_cot_seconds / num_non_cot) / num_questions))}",
            f"{(total_cot_tokens / num_cot) / num_questions:.0f}",
            f"{datetime.timedelta(seconds=((total_cot_seconds / num_cot) / num_questions))}"
        ])

        Logger.print_table(rows=table_rows, headers=["Model", "Non-CoT Gen. Tokens", "Non-CoT Exec. Time", "CoT Gen. Tokens", "CoT Exec. Time"])
