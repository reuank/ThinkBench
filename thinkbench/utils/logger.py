import datetime
from typing import List

from tabulate import tabulate

from benchmark.results import TestCaseResult
from constants import PRINT_SEPARATOR, PRINT_SEPARATOR_LENGTH, LOG_INFO, LOG_ERROR, LOGGER_PREFIX, TABLE_FORMAT


class Logger:
    @staticmethod
    def print_header(headline: str, print_out: bool = True, min_length: int = PRINT_SEPARATOR_LENGTH):
        padding = 0
        if len(headline) < min_length:
            padding = int((min_length - len(headline)) / 2)

        headline = f"{PRINT_SEPARATOR * padding} {headline} {PRINT_SEPARATOR  * padding}"

        header = "\n" + PRINT_SEPARATOR * len(headline)
        header += "\n" + headline
        header += "\n" + PRINT_SEPARATOR * len(headline)

        if print_out:
            print(header)
        else:
            return header

    @staticmethod
    def info(message: str, prefix: str = f"{LOGGER_PREFIX}Info: "):
        if LOG_INFO:
            print(f"{prefix}{message}")

    @staticmethod
    def error(message: str, prefix: str = f"{LOGGER_PREFIX}Error: "):
        if LOG_ERROR:
            print(f"{prefix}{message}")

    @staticmethod
    def print_results_table(test_case_results: List[TestCaseResult]):
        table_rows = []
        for test_case_result in test_case_results:
            table_rows.append([
                test_case_result["model"],
                f"{test_case_result['metrics']['accuracy']:.2f}",
                str(datetime.timedelta(seconds=test_case_result["execution_seconds"])).split(".")[0]  # trim ms
            ])

        Logger.print_header("Total Results")
        Logger.print_table(table_rows, ["Model", "Accuracy (%)", "Execution time"])

    @staticmethod
    def print_table(rows: List, headers: List, tablefmt: str = TABLE_FORMAT, print_out: bool = True):
        table = tabulate(tabular_data=rows, headers=headers, tablefmt=tablefmt)

        if print_out:
            print(table)
        else:
            return table

    @staticmethod
    def print_seperator(separator: str = PRINT_SEPARATOR, length: int = PRINT_SEPARATOR_LENGTH, count: int = 1, print_out: bool = True):
        output = ""

        for _ in range(count):
            output += separator * length

        if print_out:
            print(output)
        else:
            return output

    @staticmethod
    def print_headline(separator: str = PRINT_SEPARATOR, length: int = PRINT_SEPARATOR_LENGTH, count: int = 1):
        for _ in range(count):
            print(separator, length)
