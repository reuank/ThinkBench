import inspect
import json
import math
import os
from typing import List, Dict

import fire
import sklearn.metrics
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from utils import result_loader


def calcluate_stats_all_models(method: str, cot_results_dir: str, non_cot_results_dir: str):
    cot_result_files = []
    non_cot_result_files = []

    for file in sorted(os.listdir(cot_results_dir)):
        if file.endswith(".json"):
            cot_result_files.append(os.path.join(cot_results_dir, file))

    for file in sorted(os.listdir(non_cot_results_dir)):
        if file.endswith(".json"):
            non_cot_result_files.append(os.path.join(non_cot_results_dir, file))

    if len(cot_result_files) != len(non_cot_result_files):
        raise ValueError("Number of CoT and Non-CoT files are not equal.")

    cot_models = [cot_result_file.split("/")[-1].split("_")[3] for cot_result_file in cot_result_files]
    non_cot_models = [non_cot_result_file.split("/")[-1].split("_")[3] for non_cot_result_file in non_cot_result_files]

    if sorted(cot_models) != sorted(non_cot_models):
        raise ValueError("No CoT and Non-CoT result for each model.")

    stats = {}

    for model_name in cot_models:
        cot_result_file = [cot_result_file for cot_result_file in cot_result_files if model_name in cot_result_file][0]
        non_cot_result_file = [non_cot_result_file for non_cot_result_file in non_cot_result_files if model_name in non_cot_result_file][0]

        cot_result_file_data = result_loader.load_result_file(cot_result_file)
        non_cot_result_file_data = result_loader.load_result_file(non_cot_result_file)

        stats.update(
            {model_name: method_mapping[method](cot_result_file_data, non_cot_result_file_data)}
        )

    print(json.dumps(stats, indent=2))

    import csv
    with open('eggs.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        header = ["model"] + list(list(stats.values())[0].keys())
        spamwriter.writerow(header)
        for (model, result) in stats.items():
            row = [model] + list(result.values())
            spamwriter.writerow(row)

def calculate_stats_single_model(method: str, cot_result_file: str, non_cot_result_file: str):
    non_cot_result_file_data = None
    if non_cot_result_file:
        non_cot_result_file_data = result_loader.load_result_file(non_cot_result_file)

    cot_result_file_data = result_loader.load_result_file(cot_result_file)

    if method == "confidence_interval":
        confidence_intervals(cot_result_file_data)
    elif method == "confusion_matrix":
        confusion_matrix(cot_result_file_data)
    elif method == "runs_match":
        evaluate_runs_match(cot_result_file_data, non_cot_result_file_data)


def confidence_intervals(result_file_data, confidence_level: float = 0.95):
    metrics = result_file_data["metrics"]
    total_questions = metrics["total_results"]
    correct_answers = metrics["num_correct"]
    results = result_file_data["results"]

    # General info
    print_general_info(result_file_data)
    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {correct_answers}")

    # Confidence interval
    accuracy, lower_bound, upper_bound, margin = calculate_confidence_interval(correct_answers, total_questions, confidence_level)
    print("="*40)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"{confidence_level:.0%} Confidence Interval: [{lower_bound:.2%}, {upper_bound:.2%}] (± {margin:.2%})")

    # Adjusted accuracy
    observed_accuracy, adjusted_accuracy = calculate_adjusted_accuracy(correct_answers, total_questions, results)
    print("=" * 40)
    print(f"Observed Accuracy: {observed_accuracy:.2%}")
    print(f"Adjusted Accuracy: {adjusted_accuracy:.2%}")

    # plot_confusion_matrix(results)


def calculate_adjusted_accuracy(correct_answers, total_questions, results):
    observed_accuracy = correct_answers / total_questions
    chance_accuracy_total = 0

    # Berechnung der Gesamt-Chancegenauigkeit
    for result in results:
        chance_accuracy_total += 1 / len(result["labels"])

    # Durchschnittliche Chancegenauigkeit
    average_chance_accuracy = chance_accuracy_total / total_questions
    adjusted_accuracy = (observed_accuracy - average_chance_accuracy) / (1 - average_chance_accuracy)

    return observed_accuracy, adjusted_accuracy


def calculate_confidence_interval(correct_answers, total_questions, confidence_level):
    # Calculate accuracy
    p = correct_answers / total_questions

    # Calculate standard error
    SE = math.sqrt((p * (1 - p)) / total_questions)

    # Calculate z-value for the chosen level of confidence
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    # Calculate confidence interval
    margin = z * SE
    lower_bound = p - margin
    upper_bound = p + margin

    return p, lower_bound, upper_bound, margin


def confusion_matrix(results, ignore_odd_label_counts = True, ignore_number_labels = True):
    # Extract correct answers and model choices
    correct_answers = []
    model_choices = []
    odd_label_count = 0
    number_label_count = 0

    for result in results:
        if len(result["labels"]) != 4:
            odd_label_count += 1
            if ignore_odd_label_counts:
                continue

        elif result["labels"][0] == "1":
            number_label_count += 1
            if ignore_number_labels:
                continue

        correct_answers.append(result['correct_answer'])
        model_choices.append(result['model_choice'])

    print("=" * 40)
    print(f"Number of questions with label counts != 4: {odd_label_count}{' (ignored)' if ignore_odd_label_counts else ''}")
    print(f"Number of questions with number labels: {number_label_count}{' (ignored)' if ignore_number_labels else ''}")

    # Calculate the confusion matrix
    labels = sorted(set(correct_answers + model_choices))  # Ensure all possible labels are included
    conf_matrix = sklearn.metrics.confusion_matrix(correct_answers, model_choices, labels=labels)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Model Choice')
    plt.ylabel('Correct Answer')
    plt.title('Confusion Matrix')
    plt.show()


def evaluate_runs_match(cot_result_file_data, non_cot_result_file_data):
    correct_answers = get_correct_answers(cot_result_file_data)

    cot_model_choices = get_model_choices(cot_result_file_data)
    non_cot_model_choices = get_model_choices(non_cot_result_file_data)

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

    def combine_absolute_and_relative(absolute: int, relative: float):
        return f"{absolute} ({relative:.2%})"
        # return [absolute, relative]

    runs_match_result = {
        # "model": cot_result_file_data["model"],
        "results": len(cot_model_choices),
        "non_cot_uuid": non_cot_result_file_data["uuid"],
        "cot_uuid": cot_result_file_data["uuid"],
        "non_cot_accuracy": non_cot_result_file_data["metrics"]["accuracy"],
        "cot_accuracy": cot_result_file_data["metrics"]["accuracy"],
        "matches": combine_absolute_and_relative(len(matches_ids), len(matches_ids)/len(cot_model_choices)),
        "matches_correct": combine_absolute_and_relative(len(matches_correct_ids), len(matches_correct_ids)/len(matches_ids)),
        "matches_incorrect": combine_absolute_and_relative(len(matches_incorrect_ids), len(matches_incorrect_ids)/len(matches_ids)),
        "mismatches": combine_absolute_and_relative(len(mismatch_ids), len(mismatch_ids)/len(cot_model_choices)),
        "mismatches_cot_superior": combine_absolute_and_relative(len(mismatch_cot_superior_ids), len(mismatch_cot_superior_ids)/len(mismatch_ids)),
        "mismatches_non_cot_superior": combine_absolute_and_relative(len(mismatch_non_cot_superior_ids), len(mismatch_non_cot_superior_ids)/len(mismatch_ids)),
        "mismatches_none_superior": combine_absolute_and_relative(len(mismatch_none_superior_ids), len(mismatch_none_superior_ids)/len(mismatch_ids))
    }

    if print_stats:
        print(inspect.stack()[0][3])
        values = [[k, v] for (k, v) in runs_match_result.items()]
        print(tabulate(values, headers=["Key", "Value"], tablefmt="outline"))

    return runs_match_result



def get_model_choices(result_file_data: Dict) -> List[str]:
    return [single_result["model_choice"] for single_result in result_file_data["results"]]


def get_correct_answers(result_file_data: Dict) -> List[str]:
    return [single_result["correct_answer"] for single_result in result_file_data["results"]]


def print_general_info(result_file_data: Dict):
    print("=" * 40)
    print(f"Run UUID: {result_file_data['uuid']}")
    print(f"Model: {result_file_data['model']}")
    print(f"Dataset: {result_file_data['dataset_name']}")
    print(f"Benchmark: {result_file_data['benchmark_name']}, {result_file_data['use_chat_template']}, {result_file_data['label_numbering']}")


if __name__ == '__main__':
    method_mapping = {
        "runs_match": evaluate_runs_match
    }

    print_stats = False
    fire.Fire(calcluate_stats_all_models)
