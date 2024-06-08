from enum import Enum
from typing import List, Union


class TraceClass(Enum):
    TRACE_LABEL_UNAMBIGUOUS_EXTRACTION_SUCCEEDED = 1
    TRACE_LABEL_UNAMBIGUOUS_EXTRACTION_FAILED = 2
    TRACE_LABEL_AMBIGUOUS = 3
    NO_TRACE_LABEL = 4

    @staticmethod
    def get_ids() -> List[int]:
        return [trace_class.value for trace_class in TraceClass]

    @staticmethod
    def is_trace_class(class_id: Union[int, str]) -> bool:
        if class_id == "":
            class_id = 0

        return int(class_id) in TraceClass.get_ids()
