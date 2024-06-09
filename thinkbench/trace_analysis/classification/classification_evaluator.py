import time
from typing import Dict, List

import seaborn as sns
import sklearn
from matplotlib import pyplot as plt

from storage.backends.csv_file_storage import CsvFileStorage
from storage.backends.json_file_storage import JsonFileStorage
from storage.storage_backend import StorageBackend
from trace_analysis.classification.automatic_trace_classifier import TraceClass
from trace_analysis.classification.classification_result import ClassificationResult, SingleClassification
from trace_analysis.classification.trace_classifier import TraceClassifier
from utils.logger import Logger


class ClassificationEvaluator:
    @staticmethod
    def evaluate_classifications(classification_results: List[ClassificationResult]):
        json_file_storage = JsonFileStorage()
        csv_file_storage = CsvFileStorage()

        accuracy_evaluation_table_rows = []
        evaluated_models = []

        all_classifications = {
            "all_manual_class_ids": [],
            "all_automatic_class_ids": []
        }

        mismatches: Dict[str, List[SingleClassification]] = {}

        for classification_result in classification_results:
            # filter out classifications with no manual and automatic class id
            classification_result["results"] = [
                single_classification for single_classification in classification_result["results"]
                if single_classification["manual_class_id"] and single_classification["automatic_class_id"]
            ]

            manual_class_ids = TraceClassifier.get_manual_class_ids(classification_result)
            automatic_class_ids = TraceClassifier.get_automatic_class_ids(classification_result)

            accuracy = ClassificationEvaluator.calculate_percentage_match(
                array_a=manual_class_ids,
                array_b=automatic_class_ids
            )
            accuracy_evaluation_table_rows.append([classification_result["model"], len(manual_class_ids), f"{accuracy:.2f}%"])

            # TODO: Get analysis path from env, not from storage
            conf_matrix_file_name = StorageBackend.get_run_dependent_file_name(
                model_name=classification_result["model"],
                cot_uuid=classification_result["cot_uuid"],
                non_cot_uuid=classification_result["non_cot_uuid"],
                benchmark_name=classification_result["cot_benchmark_name"],
                dataset_name=classification_result["dataset_name"],
                suffix="classification_confusion_matrix",
                extension="pdf"
            )

            ClassificationEvaluator.save_confusion_matrix(
                true_classes=manual_class_ids,
                automatic_classes=automatic_class_ids,
                title=f"Confusion Matrix for model {classification_result['model']}",
                conf_matrix_file_name=conf_matrix_file_name
            )

            all_classifications["all_manual_class_ids"].extend(manual_class_ids)
            all_classifications["all_automatic_class_ids"].extend(automatic_class_ids)

            mismatches.update({classification_result["model"]: []})
            for single_classification in classification_result["results"]:
                if single_classification["manual_class_id"] != single_classification["automatic_class_id"]:
                    mismatches[classification_result["model"]].append(single_classification)

            evaluated_models.append(classification_result["model"])

        # TODO: Get analysis path from env, not from storage
        conf_matrix_file_name = StorageBackend.get_run_dependent_file_name(
            model_name="_all",
            cot_uuid=str(int(time.time() / 100)),
            non_cot_uuid="###",
            benchmark_name=classification_results[0]["cot_benchmark_name"],
            dataset_name=classification_results[0]["dataset_name"],
            suffix="classification_confusion_matrix",
            extension="pdf"
        )

        ClassificationEvaluator.save_confusion_matrix(
            true_classes=all_classifications["all_manual_class_ids"],
            automatic_classes=all_classifications["all_automatic_class_ids"],
            title="Confusion Matrix for all models",
            conf_matrix_file_name=conf_matrix_file_name
        )

        if len(all_classifications["all_manual_class_ids"]) > 0:
            overall_accuracy = ClassificationEvaluator.calculate_percentage_match(all_classifications["all_manual_class_ids"], all_classifications["all_automatic_class_ids"])
            Logger.info(f"Overall classification performance: {overall_accuracy:.2f}%")

            Logger.print_table(rows=accuracy_evaluation_table_rows, headers=["Model", "Total Manual Classifications", "Accuracy of Automatic Classification"])
            Logger.info(f"Confusion matrices for models {', '.join(evaluated_models)} were written to {json_file_storage.analysis_path}.")

            # Write mismatch file
            mismatch_header = []
            mismatch_rows = []
            for model, model_mismatches in mismatches.items():
                if len(model_mismatches) > 0:
                    mismatch_header = list(model_mismatches[0].keys())
                    mismatch_rows.extend([[model] + list(model_mismatch.values()) for model_mismatch in model_mismatches])

            if len(mismatch_rows) > 0:
                csv_file_storage.store_raw(
                    headers=["model"] + mismatch_header,
                    rows=mismatch_rows,
                    file_path=csv_file_storage.analysis_path / f"current_mismatches.csv"
                )

    @staticmethod
    def save_confusion_matrix(
            true_classes: List,
            automatic_classes: List,
            title: str = "Confusion Matrix",
            conf_matrix_file_name: str = ""
    ):
        json_file_storage = JsonFileStorage()

        conf_matrix = sklearn.metrics.confusion_matrix(true_classes, automatic_classes, labels=TraceClass.get_ids())
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=TraceClass.get_ids(),
                    yticklabels=TraceClass.get_ids(), cbar=False)
        plt.xlabel("Automatic Trace Class")
        plt.ylabel("Manual Trace Class")
        plt.title(title)

        (json_file_storage.analysis_path / "confusion_matrices").mkdir(parents=True, exist_ok=True)

        plt.savefig(json_file_storage.analysis_path / "confusion_matrices" / conf_matrix_file_name, format="pdf")

        return plt

    @staticmethod
    def calculate_percentage_match(array_a, array_b):
        if len(array_a) != len(array_b):
            raise ValueError("The arrays need to have the same length.")

        match_count = 0
        total_elements = len(array_a)

        for i in range(total_elements):
            if array_a[i] == array_b[i]:
                match_count += 1

        percentage_match = (match_count / total_elements) * 100

        return percentage_match
