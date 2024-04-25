from typing import List, Dict

import fire

from benchmark import Benchmark
from dataset import Dataset, Numbering
from inference import InferenceBackend, ModelConfig
from storage import StorageBackend, JsonFileStorage

from testcase import TestCase, TestCaseResult
from utils.timer import Timer


class ThinkBench:
    @staticmethod
    def run_benchmarks(models, datasets, inference_backend="default", benchmarks="default", limit=-1, labels="unchanged", use_chat_template=False, comment=""):
        Timer.get_instance("Run all").start_over()

        processed_arguments: Dict[str, List] = ThinkBench.process_arguments(
            models,
            datasets,
            inference_backend,
            benchmarks,
            limit,
            labels,
            use_chat_template,
            comment
        )

        inference_backend: InferenceBackend = InferenceBackend.get_by_name(processed_arguments["inference_backend"])
        storage_backend: StorageBackend = JsonFileStorage()

        cached_datasets: Dict[str, Dataset] = {}

        for model_name in processed_arguments["model_names"]:  # implement ensure_list
            print("=" * (41 + len(model_name)))
            print("=" * 10 + f" Benchmarking model {model_name} " + "=" * 10)
            print("=" * (41 + len(model_name)))

            model_config = ModelConfig.get_by_name(model_name)
            inference_backend.load_model_from_config(model_config)

            for dataset_name in processed_arguments["dataset_names"]:
                print(f"Loading dataset {dataset_name}")
                if dataset_name not in cached_datasets.keys():
                    cached_datasets[dataset_name] = Dataset.load_dataset_by_name(dataset_name)
                else:
                    print(f"Dataset {dataset_name} was already loaded previously.")

                dataset = cached_datasets[dataset_name]
                label_numbering: Numbering = Numbering(labels)

                for benchmark_name in processed_arguments["benchmark_names"]:
                    benchmark: Benchmark = Benchmark.get_by_name(benchmark_name)

                    test_case: TestCase = TestCase(dataset, limit, label_numbering, benchmark, use_chat_template)  # num-fewshot # use chat templates
                    test_case_result: TestCaseResult = inference_backend.run_test_case(test_case, comment)

                    storage_backend.store(test_case_result)

        print("=" * 45)
        print("=" * 45)
        Timer.get_instance("Run all").end()

    @staticmethod
    def process_arguments(models, datasets, inference_backend, benchmarks, limit, labels, use_chat_template, comment):
        def _ensure_list(parameter: str | List[str]) -> List[str]:
            if type(parameter) == str:
                parameter = [parameter]

            return parameter

        if models == "all-required":
            models = ModelConfig.get_all_required_names()
        elif models == "all":
            models = ModelConfig.get_all_names()

        if datasets == "all":
            datasets = Dataset.get_all_names()

        return {
            "model_names": _ensure_list(models),
            "dataset_names": _ensure_list(datasets),
            "inference_backend": inference_backend,
            "benchmark_names": _ensure_list(benchmarks),
            "limit": limit,
            "labels": labels,
            "use_chat_template": use_chat_template,
            "comment": comment
        }


if __name__ == '__main__':
    fire.Fire(ThinkBench.run_benchmarks)
