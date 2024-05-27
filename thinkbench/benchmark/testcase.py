import random

from benchmark.benchmark import Benchmark
from constants import RANDOM_DATA_SAMPLES_SEED
from dataset.dataset import Dataset
from dataset.single_data_instance import Numbering, Permutation


class TestCase:
    dataset: Dataset
    benchmark: Benchmark
    limit: int
    n_random: int
    random_seed: int
    label_numbering: Numbering
    label_permutation: Permutation
    use_chat_template: bool

    def __init__(
            self,
            dataset: Dataset,
            limit: int,
            n_random: int,
            label_numbering: Numbering = Numbering.get_default(),
            label_permutation: Permutation = Permutation.get_default(),
            benchmark: Benchmark = None,
            use_chat_template: bool = True
    ):
        self.dataset = dataset
        self.limit = min(limit, len(dataset.test_split)) if limit != -1 else len(dataset.test_split)
        self.n_random = n_random if limit == -1 else -1
        self.random_seed = RANDOM_DATA_SAMPLES_SEED if n_random != -1 else -1
        self.label_numbering = label_numbering
        self.label_permutation = label_permutation
        self.benchmark = benchmark
        self.use_chat_template = use_chat_template

    def prepare_test_dataset(self):
        if self.n_random != -1:
            random.seed(self.random_seed)
            instance_ids = []
            while len(instance_ids) < self.n_random:
                instance_ids.append(random.randint(0, len(self.dataset.test_split)))

            return [self.dataset.get_single_test_instance(instance_id)
                        .substitute_labels(self.label_numbering)
                        .permute_labels(self.label_permutation)
                    for instance_id in instance_ids]

        return [self.dataset.get_single_test_instance(instance_id)
                            .substitute_labels(self.label_numbering)
                            .permute_labels(self.label_permutation)
                for instance_id in range(self.limit)]

    def build_contexts(self):
        return [self.dataset.get_single_test_instance(idx).substitute_labels(self.label_numbering) for idx in
                range(self.limit)]

    def get_info(self):
        test_case_info = f"""=========== Running New Test Case ===========
Dataset: {self.dataset.name}
Limit: {self.limit}
Random Samples: {self.n_random if self.n_random != -1 else 'None'}
Random Seed: {self.random_seed if self.n_random != -1 else 'None'}
Label Numbering: {self.label_numbering}
Label Permutation: {self.label_permutation}
Benchmark: {self.benchmark.name}  
Use Chat Template: {self.use_chat_template}
============================================="""

        return test_case_info
