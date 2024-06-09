from typing import List


def ensure_list(parameter: str | List[str]) -> List[str]:
    if "," in parameter:
        parameter = parameter.split(",")
    if type(parameter) == str:
        parameter = [parameter]

    return parameter


def calculate_percentage_match(array_a, array_b):
    if len(array_a) != len(array_b):
        raise ValueError("The arrays need to have the same length.")

    match_count = 0
    total_elements = len(array_a)

    for i in range(total_elements):
        if array_a[i] == array_b[i]:
            match_count += 1

    percentage_match = (match_count / total_elements) * 100

    return percentage_match


def float_list_to_percent(float_list: List[float]) -> List[str]:
    return [f"{value:.2%}" if isinstance(value, float) else value for value in float_list]
