from dataset.dataset import DATASET_REGISTRY
from dataset.hf_dataset import HFDataset
from dataset.single_data_instance import SingleDataInstance


@DATASET_REGISTRY.register("logiqa2")
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

