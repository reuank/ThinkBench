import json
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict


class Numbering(Enum):
    UNCHANGED = "unchanged"
    LETTERS = "letters"
    NUMBERS = "numbers"
    ROMAN = "roman"

    @staticmethod
    def get_default():
        return Numbering.UNCHANGED


class SingleDataInstance:
    def __init__(self, id: int, row_name: str, context: str, question: str, answer_texts: List[str], answer_labels: List[str], correct_key: int):
        self.id = id
        self.row_name = row_name
        self.context = context
        self.question = question
        self.answer_texts: List[str] = answer_texts
        self.answer_labels: List[str] = answer_labels
        self.correct_key: int = correct_key

    @staticmethod
    def get_default_labels(count: int):
        if count > 26:
            raise ValueError("Label count exceeds letters in the English alphabet.")
        return [chr(idx + 65) for idx in range(count)]

    @staticmethod
    def get_dummy():
        return SingleDataInstance(
            id=100,
            row_name="row",
            context="{Context}",
            question="{Question}",
            answer_texts=["Text 1", "Text 2"],
            answer_labels=["A", "B"],
            correct_key=0
        )

    def substitute_labels(self, numbering: Numbering):
        if numbering == Numbering.UNCHANGED:
            pass
        elif numbering == Numbering.LETTERS:
            if all(label.isnumeric() for label in self.answer_labels):
                self.answer_labels = [f"{chr(64 + int(x))}" for x in self.answer_labels]  # Generate ["A", "B"] from ["1", "2"]
        elif numbering == Numbering.NUMBERS:
            self.answer_labels = [f"{x + 1}" for x in range(len(self.answer_labels))]  # Generate ["1", "2"] from ["A", "B"]
        else:
            raise ValueError(f"Numbering {numbering.value} not implemented.")

        return self

    def __repr__(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)
        # return f"""{self.id=}, {self.row_name=}, {self.context=}, {self.question=}, {self.answer_texts=}, {self.answer_labels=}, {self.correct_key=}"""


class Dataset(ABC):
    loaded_dataset = None

    @property
    def name(self):
        return self.__class__.__name__

    @staticmethod
    def load_dataset_by_name(dataset_name: str):
        return dataset_mapping[dataset_name]()

    @staticmethod
    def get_all_names():
        return dataset_mapping.keys()

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

    # def prepare_dataset(self, label_numbering: Numbering, limit: int) -> List[SingleDataInstance]:
    #    limit = min(limit, len(self.loaded_dataset)) if limit != -1 else len(self.loaded_dataset)
    #    return [self.get_single_instance(idx).substitute_labels(label_numbering) for idx in range(limit)]


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


class ARCChallengeDataset(HFDataset):
    hf_path = "ai2_arc"
    hf_name = "ARC-Challenge"
    test_split_name = "test"
    has_validation_data = True
    validation_split_name = "validation"

    def get_single_instance(self, split: str, id: int):
        dataset_row = self.loaded_dataset[split][id]

        return SingleDataInstance(
            id=id,
            row_name=str(dataset_row["id"]),
            context="",
            question=str(dataset_row["question"]),
            answer_texts=dataset_row["choices"]["text"],
            answer_labels=dataset_row["choices"]["label"],
            correct_key=dataset_row["choices"]["label"].index(dataset_row["answerKey"])
        )


class LogiQA2Dataset(HFDataset):
    hf_path = "logikon/logikon-bench"
    hf_name = "logiqa2"
    test_split_name = "test"
    has_validation_data = False
    validation_split_name = ""

    def get_single_instance(self, split: str, id: int):
        dataset_row = self.loaded_dataset[split][id]

        return SingleDataInstance(
            id=id,
            row_name="",
            context=str(dataset_row["passage"]),
            question=str(dataset_row["question"]),
            answer_texts=dataset_row["options"],
            answer_labels=SingleDataInstance.get_default_labels(len(dataset_row["options"])),
            correct_key=dataset_row["answer"]
        )


dataset_mapping: Dict[str, callable] = {
    "arc-challenge": ARCChallengeDataset,
    "logiqa2": LogiQA2Dataset
}
