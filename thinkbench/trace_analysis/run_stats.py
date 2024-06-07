from abc import ABC
from typing import Dict, List, Any

import numpy as np

from storage.backends.csv_file_storage import CsvFileStorage
from utils.logger import Logger


class RunStat(ABC):
    @staticmethod
    def get_model_choices(result_file_data: Dict) -> List[str]:
        return [single_result["model_choice"] for single_result in result_file_data["results"]]

    @staticmethod
    def get_correct_answers(result_file_data: Dict) -> List[str]:
        return [single_result["correct_answer"] for single_result in result_file_data["results"]]

    @staticmethod
    def float_list_to_percent(float_list: List[float]) -> List[str]:
        return [f"{value:.2%}" if isinstance(value, float) else value for value in float_list]

    @staticmethod
    def get_question_ids_of_class(class_id: int, cot_result, non_cot_result):
        csv_file_storage = CsvFileStorage()

        automatic_classifications_filename = csv_file_storage.get_classifications_filename(
            model_name=cot_result['model'],
            cot_uuid=cot_result['uuid'],
            non_cot_uuid=non_cot_result['uuid']
        )
        automatic_classifications = csv_file_storage.load_analysis_result(automatic_classifications_filename)

        question_ids_of_class = []
        for automatic_classification_row in automatic_classifications:
            if class_id == -1:
                question_ids_of_class.append(
                    int(automatic_classification_row["question_id"])
                )
            else:
                if class_id == int(automatic_classification_row["automatic_category_id"]):
                    question_ids_of_class.append(
                        int(automatic_classification_row["question_id"])
                    )

        return question_ids_of_class


class RunsMatchStat(RunStat):
    @staticmethod
    def compute_all(cot_results, non_cot_results, class_id: int = -1):
        for result_id in range(len(cot_results)):
            question_ids_of_class = RunStat.get_question_ids_of_class(
                class_id=class_id,
                cot_result=cot_results[result_id],
                non_cot_result=non_cot_results[result_id]
            )

            complete_result = RunsMatchStat.compute(
                cot_result_file_data=cot_results[result_id],
                non_cot_result_file_data=non_cot_results[result_id],
                question_ids_of_class=question_ids_of_class
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

            Logger.print_header(f"Runs Match Stats for model {complete_result['model']}, category: {'all' if class_id == -1 else class_id}")
            Logger.print_table(
                rows=[[row_headers[i]] + RunsMatchStat.float_list_to_percent(row) for i, row in enumerate(relative_rows)],
                headers=[""] + column_headers
            )


    @staticmethod
    def compute(cot_result_file_data, non_cot_result_file_data, question_ids_of_class: List[int]) -> Dict[str, Any]:
        correct_answers = RunStat.get_correct_answers(cot_result_file_data)

        cot_model_choices = RunStat.get_model_choices(cot_result_file_data)
        non_cot_model_choices = RunStat.get_model_choices(non_cot_result_file_data)

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
            "model": cot_result_file_data["model"],
            "num_results": len(cot_model_choices),
            "non_cot_uuid": non_cot_result_file_data["uuid"],
            "cot_uuid": cot_result_file_data["uuid"],
            "non_cot_accuracy": non_cot_result_file_data["metrics"]["accuracy"],
            "cot_accuracy": cot_result_file_data["metrics"]["accuracy"],
            "matches": len(matches_ids),
            "matches_correct": len(matches_correct_ids),
            "matches_incorrect": len(matches_incorrect_ids),
            "mismatches": len(mismatch_ids),
            "mismatches_cot_superior": len(mismatch_cot_superior_ids),
            "mismatches_non_cot_superior": len(mismatch_non_cot_superior_ids),
            "mismatches_none_superior": len(mismatch_none_superior_ids),
        }

        return runs_match_result


class ClassAccuracy(RunStat):
    @staticmethod
    def compute_all(cot_results, non_cot_results, class_id: int = -1):
        rows = []

        for result_id in range(len(cot_results)):
            question_ids_of_class = RunStat.get_question_ids_of_class(
                class_id=class_id,
                cot_result=cot_results[result_id],
                non_cot_result=non_cot_results[result_id]
            )

            complete_result = ClassAccuracy.compute(
                cot_result_file_data=cot_results[result_id],
                non_cot_result_file_data=non_cot_results[result_id],
                question_ids_of_class=question_ids_of_class
            )

            model = complete_result["model"]
            non_cot_accuracy_in_class = complete_result["correct_non_cot_results"] / complete_result["total_results_in_class"]
            cot_accuracy_in_class = complete_result["correct_cot_results"] / complete_result["total_results_in_class"]
            ratio = complete_result["total_results_in_class"] / complete_result["total_results"]

            rows.append([model, non_cot_accuracy_in_class, cot_accuracy_in_class, ratio])

        Logger.print_header(f"Category accuracy stats, category: {'all' if class_id == -1 else class_id}")
        Logger.print_table(
            rows=[RunStat.float_list_to_percent(row) for row in rows],
            headers=["Model", "Non CoT Accuracy", "CoT Accuracy", "Ratio"]
        )

    @staticmethod
    def compute(
            cot_result_file_data, non_cot_result_file_data, question_ids_of_class: List[int]
    ) -> Dict[str, Any]:
        correct_answers = ClassAccuracy.get_correct_answers(cot_result_file_data)

        cot_model_choices = RunStat.get_model_choices(cot_result_file_data)
        non_cot_model_choices = RunStat.get_model_choices(non_cot_result_file_data)

        if len(cot_model_choices) != len(non_cot_model_choices):
            raise ValueError("Lengths don't match")

        correct_cot_ids = []
        correct_non_cot_ids = []

        for index in range(len(cot_model_choices)):
            if index not in question_ids_of_class:
                continue

            if cot_model_choices[index] == correct_answers[index]:
                correct_cot_ids.append(index)

            if non_cot_model_choices[index] == correct_answers[index]:
                correct_non_cot_ids.append(index)

        runs_match_result = {
            "model": cot_result_file_data["model"],
            "total_results": len(cot_model_choices),
            "total_results_in_class": len(question_ids_of_class),
            "correct_cot_results": len(correct_cot_ids),
            "correct_non_cot_results": len(correct_non_cot_ids)
        }

        return runs_match_result
