import datetime
from typing import List

from tabulate import tabulate

from benchmark.results import TestCaseResult
from constants import PRINT_SEPARATOR, PRINT_SEPARATOR_LENGTH, LOG_INFO, LOG_ERROR


class Logger:
    @staticmethod
    def print_header(headline: str):
        header = f"{PRINT_SEPARATOR * 10} {headline} {PRINT_SEPARATOR  * 10}"

        print("\n" + PRINT_SEPARATOR * len(header))
        print(header)
        print(PRINT_SEPARATOR * len(header))

    @staticmethod
    def info(message: str):
        if LOG_INFO:
            print(f"### Info: {message}")

    @staticmethod
    def error(message: str):
        if LOG_ERROR:
            print(f"### Error: {message}")

    @staticmethod
    def print_results_table(test_case_results: List[TestCaseResult]):
        table_rows = []
        for test_case_result in test_case_results:
            table_rows.append([
                test_case_result["model"],
                f"{test_case_result['metrics']['accuracy']:.2f}",
                str(datetime.timedelta(seconds=test_case_result["execution_seconds"])).split(".")[0]  # trim ms
            ])

        Logger.print_seperator(count=2)
        Logger.print_table(table_rows, ["Model", "Accuracy (%)", "Execution time"])

    @staticmethod
    def print_table(rows: List, headers: List, tablefmt: str = "outline"):
        print(tabulate(tabular_data=rows, headers=headers, tablefmt=tablefmt))

    @staticmethod
    def print_seperator(separator: str = PRINT_SEPARATOR, length: int = PRINT_SEPARATOR_LENGTH, count: int = 1):
        for _ in range(count):
            print(separator, length)

    @staticmethod
    def print_headline(separator: str = PRINT_SEPARATOR, length: int = PRINT_SEPARATOR_LENGTH, count: int = 1):
        for _ in range(count):
            print(separator, length)
