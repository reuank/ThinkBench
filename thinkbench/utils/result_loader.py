import json
from pathlib import Path


def load_result_file(result_file: str):
    result_file_path: Path = Path(result_file)

    if result_file_path.is_dir():
        raise ValueError("Result file path is a directory.")
    elif result_file_path is None:
        raise ValueError("The result file path is invalid.")

    file = open(result_file_path)
    data = json.load(file)

    return data
