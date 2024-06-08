from typing import TypedDict, Optional, List


class SingleClassification(TypedDict):
    question_id: int
    reasoning: str

    cot_model_choice: str
    non_cot_model_choice: str

    manual_class_id: Optional[int]
    automatic_class_id: Optional[int]

    extracted_labels: Optional[List[str]]


class ClassificationResult(TypedDict):
    cot_uuid: str
    non_cot_uuid: str

    model: str
    dataset_name: str
    cot_benchmark_name: str
    non_cot_benchmark_name: str

    results: List[SingleClassification]
