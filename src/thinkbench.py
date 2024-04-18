from typing import List

import fire

from benchmark import Benchmark
from dataset import Dataset, Numbering
from inference import InferenceBackend, ModelConfig
from storage import StorageBackend, JsonFileStorage


# python3 thinkbench.py
# --inference_backend=llama-cpp-python
# --models=all
# --datasets=arc-challenge,logiqa2
# --limit=10
# --labels=unchanged
# --benchmark=score-individually
from testcase import TestCase, TestCaseResult


class ThinkBench:
    @staticmethod
    def run_benchmark(models, datasets, inference_backend="default", benchmark="default", limit=-1, labels="unchanged", use_chat_template=False, comment=""):
        inference_backend: InferenceBackend = InferenceBackend.get_by_name(inference_backend)
        storage_backend: StorageBackend = JsonFileStorage()

        if models == "all":
            models = ModelConfig.get_all_names()
        if type(models) == str:
            models = [models]
        if type(datasets) == str:
            datasets = [datasets]

        for model_name in models:  # implement ensure_list
            print("=" * (41 + len(model_name)))
            print("=" * 10 + f" Benchmarking model {model_name} " + "=" * 10)
            print("=" * (41 + len(model_name)))

            model_config = ModelConfig.get_by_name(model_name)
            inference_backend.load_model_from_config(model_config)

            for dataset_name in datasets:
                print(f"Loading dataset {dataset_name}")
                dataset: Dataset = Dataset.load_dataset_by_name(dataset_name)
                # prepared_dataset = dataset.prepare_dataset(Numbering(labels), limit)
                label_numbering: Numbering = Numbering(labels)
                benchmark: Benchmark = Benchmark.get_by_name(benchmark)

                test_case: TestCase = TestCase(dataset, limit, label_numbering, benchmark, use_chat_template)  # num-fewshot # use chat templates
                test_case_result: TestCaseResult = inference_backend.run_test_case(test_case, comment)

                storage_backend.store(test_case_result)



if __name__ == '__main__':
    fire.Fire(ThinkBench.run_benchmark)
