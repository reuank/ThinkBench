import csv
from pathlib import Path
from typing import List, Dict, Union

from benchmark.results import TestCaseResult
from constants import CSV_DELIMITER
from storage.storage_backend import StorageBackend
from utils.logger import Logger


class CsvFileStorage(StorageBackend):
    def store_test_case_result(self, test_case_result: TestCaseResult):
        raise NotImplementedError

    def store_analysis_result(self, headers: List, rows: List[List], filename: str = ""):
        self.store_raw(headers, rows, self.analysis_path / filename)

    def load_analysis_result(self, filename: Union[str, Path]) -> List[Dict]:
        file_path = self.analysis_path / filename
        if not file_path.exists() and not file_path.is_file():
            raise ValueError(f"File {filename} does not exist.")

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
