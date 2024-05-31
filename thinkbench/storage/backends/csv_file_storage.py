import csv
from pathlib import Path
from typing import List, Dict

from benchmark.results import TestCaseResult
from constants import CSV_DELIMITER
from storage.storage_backend import StorageBackend


class CsvFileStorage(StorageBackend):
    def store_test_case_result(self, test_case_result: TestCaseResult):
        raise NotImplementedError

    def store_analysis_result(self, headers: List, rows: List[List], filename: str = ""):
        # TODO: Confirm before overriding file
        self.store_raw(headers, rows, self.analysis_path / filename)

    def load_analysis_result(self, filename: str) -> List[Dict]:
        with open(self.analysis_path / filename, mode='r', newline='') as csvfile:
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
            print(f"File {file_path.name} was written to folder {file_path.parent}.")
