import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

import fire

from benchmark.testcase import TestCase
from benchmark.results import TestCaseResult
from constants import TIMER_VERBOSE, STORAGE_BACKEND, INFERENCE_BACKEND_VERBOSE, PRINT_SEPARATOR, PRINT_SEPARATOR_LENGTH
from dataset.dataset import Dataset
from dataset.single_data_instance import Numbering, Permutation
from inference.backends.llama_cpp_server_backend import LlamaCppServerInferenceBackend
from inference.inference_backend import InferenceBackend
from utils.list_utils import ensure_list
from utils.timer import Timer
from tabulate import tabulate
from model_config.model_config import ModelConfig


@dataclass
class ThinkBenchArguments:
    models: str = "default"
    datasets: str = "default"
    inference_backend: str = "default"
    benchmarks: str = "default"
    limit: int = -1
    random: int = -1
    labels: str = "unchanged"
    permutation: str = "unchanged"
    use_chat_template: bool = False
    verbose: bool = False
    comment: str = ""


class ThinkBench:
    @staticmethod
    def run_benchmarks(raw_arguments: ThinkBenchArguments):
        processed_arguments: Dict[str, Any] = {
            "model_configs": MODEL_CONFIG_REGISTRY.get_list(ensure_list(raw_arguments.models)),
            "datasets": DATASET_REGISTRY.get_list(ensure_list(raw_arguments.datasets)),
            "inference_backend": INFERENCE_BACKEND_REGISTRY.get_single(raw_arguments.inference_backend),
            "benchmarks": BENCHMARK_REGISTRY.get_list(raw_arguments.benchmarks),
            "label_numbering": Numbering(raw_arguments.labels),
            "label_permutation": Permutation(raw_arguments.permutation),
        }

        Timer.set_verbosity(TIMER_VERBOSE)
        Timer.get_instance("Run all").start_over()

        inference_backend: InferenceBackend = processed_arguments["inference_backend"]()
        inference_backend.set_verbosity(INFERENCE_BACKEND_VERBOSE)
        storage_backend = STORAGE_BACKEND_REGISTRY.get(STORAGE_BACKEND)()

        cached_datasets: Dict[str, Dataset] = {}
        metrics_list: List[List] = []

        for model_config in processed_arguments["model_configs"]:
            ThinkBench.print_model_header(model_config)
            inference_backend.load_model_from_config(model_config)

            for dataset in processed_arguments["datasets"]:
                if dataset.name not in cached_datasets.keys():
                    cached_datasets[dataset.name] = dataset()
                else:
                    print(f"Dataset {dataset.name} was already loaded previously.")

                dataset = cached_datasets[dataset.name]

                for benchmark in processed_arguments["benchmarks"]:
                    test_case: TestCase = TestCase(
                        dataset=dataset,
                        limit=raw_arguments.limit,
                        n_random=raw_arguments.random,
                        label_numbering=processed_arguments["label_numbering"],
                        label_permutation=processed_arguments["label_permutation"],
                        benchmark=benchmark(),
                        use_chat_template=raw_arguments.use_chat_template,
                    )

                    test_case_result: TestCaseResult = inference_backend.run_test_case(
                        test_case=test_case,
                        comment=raw_arguments.comment
                    )

                    metrics_list.append([
                        test_case_result["model"],
                        f"{test_case_result['metrics']['accuracy']:.2f}",
                        str(datetime.timedelta(seconds=test_case_result["execution_seconds"])).split(".")[0]
                    ])

                    storage_backend.store(test_case_result)

        ThinkBench.print_summary(metrics_list)

        Timer.get_instance("Run all").end(print_out=True)

        if isinstance(inference_backend, LlamaCppServerInferenceBackend):
            inference_backend.terminate_all_running_servers()

    @staticmethod
    def print_model_header(model_config: ModelConfig):
        header = f"{PRINT_SEPARATOR * 10} Benchmarking model {model_config.model_name} {PRINT_SEPARATOR  * 10}"
        print("\n" + PRINT_SEPARATOR * len(header))
        print(header)
        print(PRINT_SEPARATOR * len(header))

    @staticmethod
    def print_summary(metrics_list):
        print(PRINT_SEPARATOR * PRINT_SEPARATOR_LENGTH)
        print(PRINT_SEPARATOR * PRINT_SEPARATOR_LENGTH)
        print(tabulate(metrics_list, headers=["Model", "Accuracy (%)", "Execution time"], tablefmt="outline"))


if __name__ == '__main__':
    from benchmark.benchmark import BENCHMARK_REGISTRY
    from dataset.dataset import DATASET_REGISTRY
    from inference.inference_backend import INFERENCE_BACKEND_REGISTRY
    from model_config.model_config import MODEL_CONFIG_REGISTRY, ModelConfig
    from storage.storage_backend import STORAGE_BACKEND_REGISTRY

    def run_benchmarks_cli(**kwargs):
        raw_arguments = ThinkBenchArguments(**kwargs)
        ThinkBench.run_benchmarks(raw_arguments)

    fire.Fire(run_benchmarks_cli)
