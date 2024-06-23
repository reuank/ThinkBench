import csv
from pathlib import Path
from typing import List, Dict, Union

from benchmark.results import TestCaseResult
from constants import CSV_DELIMITER
from storage.storage_backend import StorageBackend
from evaluation.classification.classification_result import ClassificationResult
from utils.logger import Logger


class CsvFileStorage(StorageBackend):
    def store_classification_result(self, classification_result: ClassificationResult):
        pass

    def store_test_case_result(self, test_case_result: TestCaseResult):
        raise NotImplementedError

    def store_analysis_result(self, headers: List, rows: List[List], file_name: str = ""):
        self.store_raw(headers, rows, self.analysis_path / file_name)

    def load_analysis_result(self, file_name: Union[str, Path]) -> List[Dict]:
        file_path = self.analysis_path / file_name
        if not file_path.exists() and not file_path.is_file():
            raise ValueError(f"File {file_name} does not exist.")

        with open(file_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=CSV_DELIMITER)
            data = [row for row in reader]

        return data

    @staticmethod
    def store_raw(headers: List, rows: List[List], file_path: Path, print_path: bool = False):
        with open(file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=CSV_DELIMITER)
            csvwriter.writerow(headers)
            csvwriter.writerows(rows)

        if print_path:
            Logger.info(f"File {file_path.name} was written to folder {file_path.parent}.")
