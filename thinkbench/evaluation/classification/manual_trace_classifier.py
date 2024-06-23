from pathlib import Path
from typing import List

from benchmark.results import TestCaseResult
from storage.backends.json_file_storage import JsonFileStorage
from evaluation.classification.classification_result import ClassificationResult, SingleClassification
from evaluation.classification.interactive_classifier import InteractiveClassifier
from evaluation.classification.trace_classifier import TraceClassifier
from utils.cli_interactions import Interaction
from utils.test_case_result_helper import TestCaseResultHelper


class ManualTraceClassifier(TraceClassifier):
    @staticmethod
    def classify_test_case_results(
            cot_test_case_results: List[TestCaseResult],
            non_cot_test_case_results: List[TestCaseResult],
            override: bool
    ) -> List[ClassificationResult]:
        json_storage_backend: JsonFileStorage = JsonFileStorage()
        classification_results: List[ClassificationResult] = []

        # Loop over Test Case Results
        for cot_test_case_result_id, cot_test_case_result in enumerate(cot_test_case_results):
            classify_again = True
            non_cot_test_case_result = non_cot_test_case_results[cot_test_case_result_id]

            non_cot_model_choices = TestCaseResultHelper.get_model_choices(non_cot_test_case_result)
            cot_model_choices = TestCaseResultHelper.get_model_choices(cot_test_case_result)

            classification_result_file_name = JsonFileStorage.get_classification_result_file_name_for_test_cases(
                cot_test_case_result,
                non_cot_test_case_result
            )

            classification_result_file_path: Path = json_storage_backend.analysis_path / classification_result_file_name
            if classification_result_file_path.is_file() and not override:
                manual_classification_result = json_storage_backend.load_classification_result(
                    classification_result_file_name=classification_result_file_name
                )

                manual_class_ids = [
                    0 if not single_classification["manual_class_id"]
                    else int(single_classification["manual_class_id"])
                    for single_classification in manual_classification_result["results"]
                ]

                num_labeled = sum(manual_class_id != 0 for manual_class_id in manual_class_ids)

                classify_again = Interaction.query_yes_no(
                    question=f"A classification file for this run already exists for model {cot_test_case_result['model']}."
                             f"\nIt contains {num_labeled} manually labeled traces. Do you want to continue?",
                    default="no"
                )

            if classify_again:
                cot_result_with_samples = ManualTraceClassifier.replace_results_with_samples(
                    test_case_result=cot_test_case_result.copy(),
                    seed=cot_test_case_result["model"]
                )

                manual_classification_results: List[SingleClassification] = []

                for cot_single_benchmark_result in cot_result_with_samples["results"]:
                    if "reasoning" not in cot_single_benchmark_result["completions"][0].keys():
                        raise ValueError("CoT row does not contain a reasoning trace.")

                    manual_classification_results.append(
                        SingleClassification(
                            question_id=cot_single_benchmark_result["question_id"],
                            reasoning=cot_single_benchmark_result["completions"][0]["reasoning"]["text"].strip(),
                            non_cot_model_choice=non_cot_model_choices[cot_single_benchmark_result["question_id"]],
                            cot_model_choice=cot_model_choices[cot_single_benchmark_result["question_id"]],
                            manual_class_id=None,
                            automatic_class_id=None,
                            extracted_labels=[]
                        )
                    )

                classification_results.append(
                    ClassificationResult(
                        model=cot_test_case_result["model"],
                        non_cot_uuid=non_cot_test_case_result["uuid"],
                        cot_uuid=cot_test_case_result["uuid"],
                        dataset_name=cot_test_case_result["dataset_name"],
                        non_cot_benchmark_name=non_cot_test_case_result["benchmark_name"],
                        cot_benchmark_name=cot_test_case_result["benchmark_name"],
                        results=manual_classification_results
                    )
                )

        interactive_classifier = InteractiveClassifier()
        interactive_classifier.classify(classification_results)

        return interactive_classifier.classification_results
