from typing import List, TypedDict, Optional, Dict

from benchmark import Benchmark, SingleBenchmarkResult, Metrics
from dataset import Dataset, Numbering, SingleDataInstance


class TestCaseResult(TypedDict):
    model: str
    dataset_name: str
    benchmark_name: str
    label_numbering: str
    hostname: str
    inference_backend: str
    inference_backend_properties: Optional[Dict]
    metrics: Metrics
    start_time: float
    end_time: float
    execution_seconds: float
    comment: str
    use_chat_template: bool
    results: List[SingleBenchmarkResult]


class TestCase:
    dataset: Dataset
    benchmark: Benchmark
    limit: int
    label_numbering: Numbering
    use_chat_template: bool

    def __init__(self, dataset: Dataset, limit: int, label_numbering: Numbering = Numbering.get_default(), benchmark: Benchmark = Benchmark.get_default(), use_chat_template: bool = True):
        self.dataset = dataset
        self.limit = min(limit, len(dataset.test_split)) if limit != -1 else len(dataset.test_split)
        self.label_numbering = label_numbering
        self.benchmark = benchmark
        self.use_chat_template = use_chat_template

    def prepare_test_dataset(self):
        return [self.dataset.get_single_test_instance(idx).substitute_labels(self.label_numbering) for idx in range(self.limit)]

    def build_contexts(self):
        return [self.dataset.get_single_test_instance(idx).substitute_labels(self.label_numbering) for idx in
                range(self.limit)]

    def get_info(self):
        test_case_info = f"""=========== Running New Test Case ===========
Dataset: {self.dataset.name}
Limit: {self.limit}
Label Numbering: {self.label_numbering}
Benchmark: {self.benchmark.name}  
Use Chat Template: {self.use_chat_template}
============================================="""

        return test_case_info