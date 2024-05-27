from abc import abstractmethod, ABC

from dataset.single_data_instance import SingleDataInstance
from utils.registry import Registry


class Dataset(ABC):
    loaded_dataset = None

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def load_dataset(self):
        raise NotImplementedError

    # ===============================================
    # Test Data
    # ===============================================
    @property
    @abstractmethod
    def test_split_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def test_split(self):
        raise NotImplementedError

    @abstractmethod
    def get_single_test_instance(self, id: int) -> SingleDataInstance:
        raise NotImplementedError

    # ===============================================
    # Validation Data (e.g. for few-shot examples)
    # ===============================================
    @property
    @abstractmethod
    def has_validation_data(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def validation_split_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def validation_split(self):
        raise NotImplementedError

    @abstractmethod
    def get_single_validation_instance(self, id: int) -> SingleDataInstance:
        raise NotImplementedError


DATASET_REGISTRY = Registry(
    registry_name="dataset",
    base_class=Dataset,
    lazy_load_dirs=["dataset/datasets"]
)