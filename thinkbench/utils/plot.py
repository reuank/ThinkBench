from typing import List

import seaborn as sns
import sklearn
from matplotlib import pyplot as plt

from storage.backends.json_file_storage import JsonFileStorage


class Plot:
    @staticmethod
    def save_confusion_matrix(
            true_classes: List,
            automatic_classes: List,
            all_classes: List,
            x_label: str,
            y_label: str,
            sub_folder: str = ".",
            title: str = "Confusion Matrix",
            conf_matrix_file_name: str = ""
    ):
        json_file_storage = JsonFileStorage()

        conf_matrix = sklearn.metrics.confusion_matrix(true_classes, automatic_classes, labels=all_classes)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=all_classes, yticklabels=all_classes, cbar=False)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        (json_file_storage.analysis_path / "confusion_matrices" / sub_folder).mkdir(parents=True, exist_ok=True)

        plt.savefig(json_file_storage.analysis_path / "confusion_matrices" / sub_folder / conf_matrix_file_name, format="pdf")

        return plt