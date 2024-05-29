from abc import ABC, abstractmethod
from typing import List

from benchmark.results import TestCaseResult
from utils.registry import Registry


class StorageBackend(ABC):
    @abstractmethod
    def store(self, test_case_result: TestCaseResult):
        raise NotImplementedError

    def store_multiple(self, test_case_result_list: List[TestCaseResult]):
        for test_case_result in test_case_result_list:
            self.store(test_case_result)


STORAGE_BACKEND_REGISTRY = Registry(
    registry_name="storage",
    base_class=StorageBackend,
    lazy_load_dirs=["storage/backends"]
)
