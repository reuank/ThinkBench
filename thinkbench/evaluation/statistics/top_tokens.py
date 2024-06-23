from typing import List, Dict, Tuple

from benchmark.results import TestCaseResult
from evaluation.statistics.run_stat import RunStat
from utils.logger import Logger
from utils.test_case_result_helper import TestCaseResultHelper


class TopTokens(RunStat):
    @staticmethod
    def compute_all(
            cot_test_case_results: List[TestCaseResult],
            non_cot_test_case_results: List[TestCaseResult],
            class_id: int = -1,
            **kwargs
    ):
        run: str = kwargs.get("run", "cot")
        class_part: str = kwargs.get("class_part", "all_in_class")

        if run == "non-cot":
            analyzed_test_case_results = non_cot_test_case_results
        else:
            analyzed_test_case_results = cot_test_case_results

        models = []
        top_token_columns = []

        for test_case_result_id, test_case_result in enumerate(analyzed_test_case_results):
            cot_label_completion_top_tokens, non_cot_label_completion_top_tokens = TopTokens.compute(
                cot_test_case_result=cot_test_case_results[test_case_result_id],
                non_cot_test_case_result=non_cot_test_case_results[test_case_result_id],
                class_id=class_id,
                class_part=class_part
            )

            models.append(test_case_result['model'])
            
            top_token_column = []
            if run == "non-cot":
                analyzed_label_completion_top_tokens = non_cot_label_completion_top_tokens
            else:
                analyzed_label_completion_top_tokens = cot_label_completion_top_tokens

            for token_id, token_frequency in enumerate(analyzed_label_completion_top_tokens):
                top_token_column.append(f"{repr(token_frequency[0])} ({token_frequency[1]})")

            top_token_columns.append(top_token_column)

        Logger.print_header(f"Benchmark: {analyzed_test_case_results[0]['benchmark_name']}, "
                            f"Dataset: {analyzed_test_case_results[0]['dataset_name']}")
        Logger.print_table(rows=list(zip(*top_token_columns))[:10], headers=models)

    @staticmethod
    def compute(
            cot_test_case_result: TestCaseResult,
            non_cot_test_case_result: TestCaseResult,
            class_id: int = -1,
            class_part: str = "all_in_class"
    ) -> (List, List):
        cot_indexes_to_keep, non_cot_indexes_to_keep = RunStat.get_class_part_question_ids(
            cot_test_case_result=cot_test_case_result,
            non_cot_test_case_result=non_cot_test_case_result,
            class_id=class_id,
            class_part=class_part
        )

        cot_top_tokens: Dict[str, List[int]] = TestCaseResultHelper.get_label_completion_top_tokens(cot_test_case_result)
        non_cot_top_tokens: Dict[str, List[int]] = TestCaseResultHelper.get_label_completion_top_tokens(non_cot_test_case_result)

        sorted_cot_top_token_frequencies = TopTokens.get_sorted_token_frequencies(cot_top_tokens, non_cot_indexes_to_keep)
        sorted_non_cot_top_token_frequencies = TopTokens.get_sorted_token_frequencies(non_cot_top_tokens, non_cot_indexes_to_keep)

        return sorted_cot_top_token_frequencies, sorted_non_cot_top_token_frequencies

    @staticmethod
    def get_sorted_token_frequencies(top_tokens: Dict[str, List[int]], indexes_to_keep: List[int]) -> List[Tuple[str, int]]:
        top_tokens = top_tokens.copy()
        for top_token, top_token_question_ids in top_tokens.items():
            for top_token_question_id in top_token_question_ids:
                if indexes_to_keep and top_token_question_id not in indexes_to_keep:
                    top_tokens[top_token].remove(top_token_question_id)

        top_token_frequencies: Dict[str, int] = {token: len(question_ids) for token, question_ids in top_tokens.items()}
        sorted_top_token_frequencies = sorted(top_token_frequencies.items(), key=lambda item: item[1], reverse=True)

        return sorted_top_token_frequencies