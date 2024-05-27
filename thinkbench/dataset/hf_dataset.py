import time
from abc import abstractmethod, ABC

from dataset.dataset import Dataset
from dataset.single_data_instance import SingleDataInstance


class HFDataset(Dataset, ABC):
    @property
    @abstractmethod
    def hf_path(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def hf_name(self) -> str:
        raise NotImplementedError

    def __init__(self):
        self.load_dataset()

    def load_dataset(self):
        import datasets as hf_datasets

        start_dataset_load = time.time()

        # hf_datasets.logging.set_verbosity_info()
        dataset = hf_datasets.load_dataset(
            path=self.hf_path,
            name=self.hf_name
        )

        end_dataset_load = time.time()

        print(
            f"Dataset {self.hf_name} loaded in {round(end_dataset_load - start_dataset_load, 2)} seconds")

        self.loaded_dataset = dataset

    @property
    def validation_split(self):
        return self.loaded_dataset[self.validation_split_name]

    @property
    def test_split(self):
        return self.loaded_dataset[self.test_split_name]

    def get_single_test_instance(self, id: int) -> SingleDataInstance:
        return self.get_single_instance(self.test_split_name, id)

    def get_single_validation_instance(self, id: int) -> SingleDataInstance:
        if self.has_validation_data:
            return self.get_single_instance(self.validation_split_name, id)
        else:
            raise NotImplementedError

    @abstractmethod
    def get_single_instance(self, split: str, id: int):
        raise NotImplementedError
