from abc import ABC, abstractmethod
from typing import List

from benchmark.results import TestCaseResult
from utils.registry import Registry


class StorageBackend(ABC):
    @abstractmethod
    def store_test_case_result(self, test_case_result: TestCaseResult):
        raise NotImplementedError

    def store_multiple_test_case_results(self, test_case_results: List[TestCaseResult]):
        for test_case_result in test_case_results:
            self.store_test_case_result(test_case_result)


STORAGE_BACKEND_REGISTRY = Registry(
    registry_name="storage",
    base_class=StorageBackend,
    lazy_load_dirs=["storage/backends"]
)
