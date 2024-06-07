from abc import ABC
from pathlib import Path
from typing import Dict

from storage.backends.csv_file_storage import CsvFileStorage
from utils.cli_interactions import Interaction
from utils.logger import Logger


class TraceClassifier(ABC):
    @staticmethod
    def store_classification_results(classification_results: Dict[str, Dict]):
        csv_file_storage = CsvFileStorage()
        written_file_names = []

        for model, classification_result in classification_results.items():
            override = True

            classifications_file_name = csv_file_storage.get_classifications_file_name(
                model_name=model,
                cot_uuid=classification_result["cot_uuid"],
                non_cot_uuid=classification_result["non_cot_uuid"]
            )

            classifications_file_path: Path = csv_file_storage.analysis_path / classifications_file_name
            if classifications_file_path.is_file():
                override = Interaction.query_yes_no(
                    question=f"A classification file {classifications_file_name} already exists for model {model}."
                             f"\nDo you want to override it?",
                    default="yes"
                )

            if override:
                csv_file_storage.store_analysis_result(
                    headers=classification_result["result_rows"][0].keys(),
                    rows=[row.values() for row in classification_result["result_rows"]],
                    file_name=classifications_file_name
                )

                written_file_names.append(classifications_file_name)

        if written_file_names:
            Logger.info(f"Automatic classification files were written to {csv_file_storage.analysis_path}")
