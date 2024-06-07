import time
from typing import Dict

import seaborn as sns
import sklearn
from matplotlib import pyplot as plt

from storage.backends.csv_file_storage import CsvFileStorage
from trace_analysis.automatic_trace_classifier import AutomaticTraceClassifier, TraceClass
from utils.logger import Logger


class ClassificationEvaluator:
    @staticmethod
    def evaluate_classifications(classification_results: Dict[str, Dict]):
        csv_file_storage = CsvFileStorage()
        accuracy_evaluation_rows = []
        evaluated_models = []
        all_classifications = {
            "all_manual_class_ids": [],
            "all_automatic_class_ids": []
        }

        for model, classification_result in classification_results.items():
            manual_classifications_file_name = csv_file_storage.get_samples_file_name(
                model_name=model,
                cot_uuid=classification_result["cot_uuid"],
                non_cot_uuid=classification_result["non_cot_uuid"]
            )
            manual_classification_file_rows = csv_file_storage.load_analysis_result(manual_classifications_file_name)

            manual_class_ids = []
            automatic_class_ids = []

            for manual_classification_file_row in manual_classification_file_rows:
                manual_class_id = int(manual_classification_file_row["manual_class_id"])

                if manual_class_id != 0:
                    question_id = int(manual_classification_file_row["question_id"])
                    manual_class_ids.append(manual_class_id)
                    automatic_class_ids.append(
                        int(classification_result["result_rows"][question_id]["automatic_class_id"])
                    )

            num_unlabeled = len(manual_classification_file_rows) - len(manual_class_ids)

            if num_unlabeled > 0:
                Logger.info(f"There are {num_unlabeled} unlabeled samples in the manual classification file {manual_classifications_file_name}. "
                            f"Please make sure to manually label all samples to run an evaluation.")
            else:
                accuracy = ClassificationEvaluator.calculate_percentage_match(
                    array_a=manual_class_ids,
                    array_b=automatic_class_ids
                )
                accuracy_evaluation_rows.append([model, len(manual_class_ids), f"{accuracy:.2f}%"])

                conf_matrix = sklearn.metrics.confusion_matrix(manual_class_ids, automatic_class_ids, labels=TraceClass.get_ids())
                plt.figure(figsize=(10, 7))
                sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=TraceClass.get_ids(), yticklabels=TraceClass.get_ids(), cbar=False)
                plt.xlabel("Automatic Trace Class")
                plt.ylabel("Manual Trace Class")
                plt.title(f"Confusion Matrix for model {model}")

                # TODO: Get analysis path from env, not from storage
                conf_matrix_file_name = csv_file_storage.get_run_dependant_file_name(
                    model_name=model,
                    cot_uuid=manual_classification_file_rows[0]["cot_uuid"],
                    non_cot_uuid=manual_classification_file_rows[0]["non_cot_uuid"],
                    suffix="classification_confusion_matrix",
                    extension="pdf"
                )
                plt.savefig(csv_file_storage.analysis_path / conf_matrix_file_name, format="pdf")

                all_classifications["all_manual_class_ids"].extend(manual_class_ids)
                all_classifications["all_automatic_class_ids"].extend(automatic_class_ids)
                evaluated_models.append(model)

        conf_matrix = sklearn.metrics.confusion_matrix(all_classifications["all_manual_class_ids"], all_classifications["all_automatic_class_ids"], labels=TraceClass.get_ids())
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=TraceClass.get_ids(), yticklabels=TraceClass.get_ids(), cbar=False)
        plt.xlabel("Automatic Trace Class")
        plt.ylabel("Manual Trace Class")
        plt.title(f"Confusion Matrix for all models")

        # TODO: Get analysis path from env, not from storage
        conf_matrix_file_name = csv_file_storage.get_run_dependant_file_name(
            model_name="all",
            cot_uuid=str(int(time.time() / 100)),
            non_cot_uuid="###",
            suffix="classification_confusion_matrix",
            extension="pdf"
        )
        plt.savefig(csv_file_storage.analysis_path / conf_matrix_file_name, format="pdf")

        if len(all_classifications["all_manual_class_ids"]) > 0:
            overall_accuracy = ClassificationEvaluator.calculate_percentage_match(all_classifications["all_manual_class_ids"], all_classifications["all_automatic_class_ids"])
            Logger.info(f"Overall classification performance: {overall_accuracy:.2f}%")

            Logger.print_table(rows=accuracy_evaluation_rows, headers=["Model", "Total Manual Classifications", "Accuracy of Automatic Classification"])
            Logger.info(f"Confusion matrices for models {', '.join(evaluated_models)} were written to {csv_file_storage.analysis_path}.")

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