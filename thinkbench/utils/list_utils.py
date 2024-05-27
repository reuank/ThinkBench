from typing import List


def ensure_list(parameter: str | List[str]) -> List[str]:
    if "," in parameter:
        parameter = parameter.split(",")
    if type(parameter) == str:
        parameter = [parameter]

    return parameter
