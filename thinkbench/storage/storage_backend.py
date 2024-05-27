from abc import ABC, abstractmethod

from benchmark.results import TestCaseResult
from utils.registry import Registry


class StorageBackend(ABC):
    @abstractmethod
    def store(self, test_case_result: TestCaseResult):
        raise NotImplementedError


STORAGE_BACKEND_REGISTRY = Registry(
    registry_name="storage",
    base_class=StorageBackend,
    lazy_load_dirs=["storage/backends"]
)
