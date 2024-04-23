from typing import List, Dict

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
    def run_benchmark(models, datasets, inference_backend="default", benchmarks="default", limit=-1, labels="unchanged", use_chat_template=False, comment=""):
        inference_backend: InferenceBackend = InferenceBackend.get_by_name(inference_backend)
        storage_backend: StorageBackend = JsonFileStorage()

        if models == "all-required":
            models = ModelConfig.get_all_required_names()
        elif models == "all":
            models = ModelConfig.get_all_names()

        if datasets == "all":
            datasets = Dataset.get_all_names()

        loaded_datasets: Dict[str, Dataset] = {}

        for model_name in ThinkBench.ensure_list(models):  # implement ensure_list
            print("=" * (41 + len(model_name)))
            print("=" * 10 + f" Benchmarking model {model_name} " + "=" * 10)
            print("=" * (41 + len(model_name)))

            model_config = ModelConfig.get_by_name(model_name)
            inference_backend.load_model_from_config(model_config)

            for dataset_name in ThinkBench.ensure_list(datasets):
                print(f"Loading dataset {dataset_name}")
                if dataset_name not in loaded_datasets.keys():
                    loaded_datasets[dataset_name] = Dataset.load_dataset_by_name(dataset_name)
                else:
                    print(f"Dataset {dataset_name} was already loaded previously.")

                dataset = loaded_datasets[dataset_name]

                label_numbering: Numbering = Numbering(labels)

                for benchmark_name in ThinkBench.ensure_list(benchmarks):
                    benchmark: Benchmark = Benchmark.get_by_name(benchmark_name)

                    test_case: TestCase = TestCase(dataset, limit, label_numbering, benchmark, use_chat_template)  # num-fewshot # use chat templates
                    test_case_result: TestCaseResult = inference_backend.run_test_case(test_case, comment)

                    storage_backend.store(test_case_result)

    @staticmethod
    def ensure_list(parameter: str | List[str]) -> List[str]:
        if type(parameter) == str:
            parameter = [parameter]

        return parameter

if __name__ == '__main__':
    fire.Fire(ThinkBench.run_benchmark)
