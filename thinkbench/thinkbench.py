import datetime
from typing import List, Dict, Any

import fire

from benchmark.benchmark import Benchmark
from benchmark.testcase import TestCase
from benchmark.results import TestCaseResult
from constants import TIMER_VERBOSE, STORAGE_BACKEND, INFERENCE_BACKEND_VERBOSE, PRINT_SEPARATOR, PRINT_SEPARATOR_LENGTH
from dataset.dataset import Dataset
from dataset.single_data_instance import Numbering, Permutation
from inference.backends.llama_cpp_server_backend import LlamaCppServerInferenceBackend
from inference.inference_backend import InferenceBackend
from utils.timer import Timer
from tabulate import tabulate


class ThinkBench:
    @staticmethod
    def run_benchmarks(
            models,
            datasets,
            inference_backend="default",
            benchmarks="default",
            limit=-1,
            random=-1,
            labels="unchanged",
            permutation="unchanged",
            use_chat_template=False,
            verbose=False,
            comment=""
    ):
        models = ThinkBench._process_models(models)
        datasets = ThinkBench._process_datasets(datasets)

        processed_arguments: Dict[str, Any] = ThinkBench._prepare_arguments(
            models,
            datasets,
            inference_backend,
            benchmarks,
            limit,
            random,
            labels,
            permutation,
            use_chat_template,
            verbose,
            comment
        )

        Timer.set_verbosity(TIMER_VERBOSE)
        Timer.get_instance("Run all").start_over()

        inference_backend = ThinkBench._setup_inference_backend(processed_arguments["inference_backend"])
        storage_backend = ThinkBench._setup_storage_backend()

        cached_datasets: Dict[str, Dataset] = {}
        metrics_list: List[List] = []

        for model_name in processed_arguments["model_names"]:
            ThinkBench._print_model_header(model_name)
            model_config = MODEL_CONFIG_REGISTRY.get(model_name)
            inference_backend.load_model_from_config(model_config)

            for dataset_name in processed_arguments["dataset_names"]:
                dataset = ThinkBench._get_or_load_dataset(dataset_name, cached_datasets)
                label_numbering = Numbering(labels)
                label_permutation = Permutation(permutation)

                for benchmark_name in processed_arguments["benchmark_names"]:
                    metrics_list.append(
                        ThinkBench._run_benchmark(
                            benchmark_name,
                            dataset,
                            inference_backend,
                            storage_backend,
                            limit,
                            random,
                            label_numbering,
                            label_permutation,
                            use_chat_template,
                            comment)
                    )

        ThinkBench._print_summary(metrics_list)

        Timer.get_instance("Run all").end(print_out=True)

        if isinstance(inference_backend, LlamaCppServerInferenceBackend):
            inference_backend.terminate_all_running_servers()

    @staticmethod
    def _process_models(models):
        if models == "all-required":
            return MODEL_CONFIG_REGISTRY.get_all_with_flag("required")
        elif models == "all":
            return list(MODEL_CONFIG_REGISTRY.keys())
        return models

    @staticmethod
    def _process_datasets(datasets):
        if datasets == "all":
            return list(DATASET_REGISTRY.keys())
        return datasets

    @staticmethod
    def _ensure_list(parameter: str | List[str]) -> List[str]:
        if "," in parameter:
            parameter = parameter.split(",")
        if type(parameter) == str:
            parameter = [parameter]

        return parameter


    @staticmethod
    def _prepare_arguments(
            models,
            datasets,
            inference_backend,
            benchmarks,
            limit,
            random,
            labels,
            permutation,
            use_chat_template,
            verbose,
            comment
    ):
        return {
            "model_names": ensure_list(models),
            "dataset_names": ensure_list(datasets),
            "inference_backend": inference_backend,
            "benchmark_names": ensure_list(benchmarks),
            "limit": limit,
            "random": random,
            "labels": labels,
            "permutation": permutation,
            "use_chat_template": use_chat_template,
            "verbose": verbose,
            "comment": comment
        }

    @staticmethod
    def _setup_inference_backend(inference_backend_name):
        inference_backend: InferenceBackend = INFERENCE_BACKEND_REGISTRY.get(inference_backend_name)()
        inference_backend.set_verbosity(INFERENCE_BACKEND_VERBOSE)
        return inference_backend

    @staticmethod
    def _setup_storage_backend():
        return STORAGE_BACKEND_REGISTRY.get(STORAGE_BACKEND)()

    @staticmethod
    def _print_model_header(model_name):
        header = f"{PRINT_SEPARATOR * 10} Benchmarking model {model_name} {PRINT_SEPARATOR  * 10}"
        print("\n" + PRINT_SEPARATOR * len(header))
        print(header)
        print(PRINT_SEPARATOR * len(header))

    @staticmethod
    def _get_or_load_dataset(dataset_name, cached_datasets):
        if dataset_name not in cached_datasets:
            print(f"Loading dataset {dataset_name}")
            cached_datasets[dataset_name] = DATASET_REGISTRY.get(dataset_name)()
        else:
            print(f"Dataset {dataset_name} was already loaded previously.")
        return cached_datasets[dataset_name]

    @staticmethod
    def _run_benchmark(
            benchmark_name,
            dataset,
            inference_backend,
            storage_backend,
            limit,
            random,
            label_numbering,
            label_permutation,
            use_chat_template,
            comment
    ):
        benchmark: Benchmark = BENCHMARK_REGISTRY.get(benchmark_name)()
        test_case = TestCase(dataset, limit, random, label_numbering, label_permutation, benchmark, use_chat_template)
        test_case_result: TestCaseResult = inference_backend.run_test_case(test_case, comment)

        metrics = [
            test_case_result["model"],
            f"{test_case_result['metrics']['accuracy']:.2f}",
            str(datetime.timedelta(seconds=test_case_result["execution_seconds"])).split(".")[0]
        ]

        storage_backend.store(test_case_result)
        return metrics

    @staticmethod
    def _print_summary(metrics_list):
        print(PRINT_SEPARATOR * PRINT_SEPARATOR_LENGTH)
        print(PRINT_SEPARATOR * PRINT_SEPARATOR_LENGTH)
        print(tabulate(metrics_list, headers=["Model", "Accuracy (%)", "Execution time"], tablefmt="outline"))


if __name__ == '__main__':
    from benchmark.benchmark import BENCHMARK_REGISTRY
    from dataset.dataset import DATASET_REGISTRY
    from inference.inference_backend import INFERENCE_BACKEND_REGISTRY
    from model_config.model_config import MODEL_CONFIG_REGISTRY
    from storage.storage_backend import STORAGE_BACKEND_REGISTRY

    fire.Fire(ThinkBench.run_benchmarks)
