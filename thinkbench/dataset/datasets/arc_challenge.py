from dataset.dataset import DATASET_REGISTRY
from dataset.hf_dataset import HFDataset
from dataset.single_data_instance import SingleDataInstance


@DATASET_REGISTRY.register(name="arc-challenge")
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
