import math

import fire
import sklearn.metrics
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from utils import result_loader


def calculate_for_file(result_file: str, confidence_level: float = 0.95):
    result_file_data = result_loader.load_result_file(result_file)
    metrics = result_file_data["metrics"]
    total_questions = metrics["total_results"]
    correct_answers = metrics["num_correct"]
    results = result_file_data["results"]

    # General info
    print("=" * 40)
    print(f"Model: {result_file_data['model']}")
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

    plot_confusion_matrix(results)


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


def plot_confusion_matrix(results, ignore_odd_label_counts = True):
    # Extract correct answers and model choices
    correct_answers = []
    model_choices = []

    for result in results:
        if ignore_odd_label_counts and len(result["labels"]) != 4:
            continue

        correct_answers.append(result['correct_answer'])
        model_choices.append(result['model_choice'])

    # Calculate the confusion matrix
    labels = sorted(set(correct_answers + model_choices))  # Ensure all possible labels are included
    conf_matrix = sklearn.metrics.confusion_matrix(correct_answers, model_choices, labels=labels, normalize='true')

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Model Choice')
    plt.ylabel('Correct Answer')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    fire.Fire(calculate_for_file)
