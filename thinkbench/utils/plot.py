from typing import List

import seaborn as sns
import sklearn
from matplotlib import pyplot as plt

from constants import BIG_PLOTS
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

        if BIG_PLOTS:
            from matplotlib import font_manager

            font_path = '/Users/leonknauer/Downloads/Source_Sans_3/static/SourceSans3-Regular.ttf'
            font_manager.fontManager.addfont(font_path)
            custom_font = font_manager.FontProperties(fname=font_path)

            plt.rcParams['font.family'] = custom_font.get_name()
            plt.rcParams['font.size'] = 22

            fig, ax = (plt.subplots(figsize=(7, 6.5)))
            sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=all_classes, yticklabels=all_classes, cbar=False, ax=ax)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            plt.subplots_adjust(top=1, right=1)
        else:
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=all_classes, yticklabels=all_classes, cbar=False)

            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)

        (json_file_storage.analysis_path / "confusion_matrices" / sub_folder).mkdir(parents=True, exist_ok=True)

        plt.savefig(json_file_storage.analysis_path / "confusion_matrices" / sub_folder / conf_matrix_file_name, format="pdf")

        return plt
