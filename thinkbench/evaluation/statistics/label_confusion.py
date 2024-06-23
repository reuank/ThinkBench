from typing import List

from benchmark.results import TestCaseResult
from storage.storage_backend import StorageBackend
from evaluation.statistics.run_stat import RunStat
from utils.logger import Logger
from utils.plot import Plot


class LabelConfusion(RunStat):
    @staticmethod
    def compute_all(
            cot_test_case_results: List[TestCaseResult],
            non_cot_test_case_results: List[TestCaseResult],
            class_id: int = -1,
            **kwargs
    ):
        Logger.info("Generating label confusion matrices for all results.")

        ignore_label_edge_cases: bool = kwargs.get('ignore_label_edge_cases', True)
        run: str = kwargs.get('run', 'cot')

        if run == "non-cot":
            analyzed_test_case_results = non_cot_test_case_results
        else:
            analyzed_test_case_results = cot_test_case_results

        for test_case_result_id, test_case_result in enumerate(analyzed_test_case_results):
            correct_answers, model_choices = LabelConfusion.get_choices(
                test_case_result=test_case_result,
                ignore_none=ignore_label_edge_cases,
                ignore_number_labels=True,
                ignore_odd_label_counts=True
            )

            conf_matrix_file_name = StorageBackend.get_run_dependent_file_name(
                model_name=test_case_result["model"],
                cot_uuid=cot_test_case_results[test_case_result_id]["uuid"],
                non_cot_uuid=non_cot_test_case_results[test_case_result_id]["uuid"],
                benchmark_name=test_case_result["benchmark_name"],
                dataset_name=test_case_result["dataset_name"],
                prefix="",
                suffix="_label_confusion_matrix",
                extension="pdf"
            )

            Plot.save_confusion_matrix(
                true_classes=correct_answers,
                automatic_classes=model_choices,
                all_classes=sorted(set(correct_answers + model_choices)),
                x_label="Model Choices",
                y_label="Correct Answers",
                title=f"Label Confusion Matrix for: {test_case_result['model']}, "
                      f"{test_case_result['dataset_name']}, {test_case_result['benchmark_name']}",
                conf_matrix_file_name=conf_matrix_file_name,
                sub_folder="label_confusion"
            )

    @staticmethod
    def get_choices(
            test_case_result: TestCaseResult,
            ignore_odd_label_counts: bool = True,
            ignore_number_labels: bool = True,
            ignore_none: bool = True
    ) -> (List, List):
        correct_answers = []
        model_choices = []

        odd_label_count = 0
        number_label_count = 0
        none_count = 0

        for single_benchmark_result in test_case_result["results"]:
            if len(single_benchmark_result["labels"]) != 4:
                odd_label_count += 1
                if ignore_odd_label_counts:
                    continue

            elif single_benchmark_result["labels"][0] == "1":
                number_label_count += 1
                if ignore_number_labels:
                    continue

            elif single_benchmark_result["model_choice"] == "NONE":
                none_count += 1
                if ignore_none:
                    continue

            correct_answers.append(single_benchmark_result['correct_answer'])
            model_choices.append(single_benchmark_result['model_choice'])

        return correct_answers, model_choices
