import datetime
from typing import List, Dict

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
from storage.storage_backend import StorageBackend
from tabulate import tabulate
from model_config.model_config import ModelConfig


class ThinkBenchArguments:
    def __init__(
        self,
        models: str = "default",
        datasets: str = "default",
        inference_backend: str = "default",
        benchmarks: str = "default",
        limit: int = -1,
        random: int = -1,
        labels: str = "unchanged",
        permutation: str = "unchanged",
        use_chat_template: bool = False,
        comment: str = ""
    ):
        self.model_configs = MODEL_CONFIG_REGISTRY.get_list(ensure_list(models))
        self.datasets = DATASET_REGISTRY.get_list(ensure_list(datasets))
        self.inference_backend = INFERENCE_BACKEND_REGISTRY.get_single(inference_backend)
        self.benchmarks = BENCHMARK_REGISTRY.get_list(benchmarks)
        self.label_numbering = Numbering(labels)
        self.label_permutation = Permutation(permutation)

        self.limit = limit
        self.n_random = random
        self.use_chat_template = use_chat_template
        self.comment = comment


class ThinkBench:
    @staticmethod
    def run_benchmarks(arguments: ThinkBenchArguments) -> None:
        inference_backend = None
        metrics_list = []

        try:
            Timer.set_verbosity(TIMER_VERBOSE)
            Timer.get_instance("Run all").start_over()

            inference_backend, storage_backend, cached_datasets = ThinkBench.setup_environment(arguments)
            metrics_list = ThinkBench.execute_benchmarks(arguments, inference_backend, storage_backend, cached_datasets)
        except Exception as e:
            print(f"An error occurred during the benchmarking process: {e}")
        finally:
            Timer.get_instance("Run all").end(print_timer=True)
            if metrics_list:
                ThinkBench.print_summary(metrics_list)
            if isinstance(inference_backend, LlamaCppServerInferenceBackend):
                inference_backend.terminate_all_running_servers()

    @staticmethod
    def execute_benchmarks(
            arguments: ThinkBenchArguments,
            inference_backend: InferenceBackend,
            storage_backend: StorageBackend,
            cached_datasets: Dict[str, Dataset]
    ) -> List[List]:
        metrics_list: List[List] = []

        for model_config in arguments.model_configs:
            ThinkBench.print_model_header(model_config)
            inference_backend.load_model_from_config(model_config)

            for dataset in arguments.datasets:
                if dataset.name not in cached_datasets.keys():
                    cached_datasets[dataset.name] = dataset()
                else:
                    print(f"Dataset {dataset.name} was already loaded previously.")

                dataset = cached_datasets[dataset.name]

                for benchmark in arguments.benchmarks:
                    test_case: TestCase = TestCase(
                        dataset=dataset,
                        limit=arguments.limit,
                        n_random=arguments.n_random,
                        label_numbering=arguments.label_numbering,
                        label_permutation=arguments.label_permutation,
                        benchmark=benchmark(),
                        use_chat_template=arguments.use_chat_template,
                    )

                    test_case_result: TestCaseResult = inference_backend.run_test_case(
                        test_case=test_case,
                        comment=arguments.comment
                    )

                    metrics_list.append([
                        test_case_result["model"],
                        f"{test_case_result['metrics']['accuracy']:.2f}",
                        str(datetime.timedelta(seconds=test_case_result["execution_seconds"])).split(".")[0]  # trim ms
                    ])

                    storage_backend.store(test_case_result)

        return metrics_list

    @staticmethod
    def setup_environment(arguments: ThinkBenchArguments) -> (InferenceBackend, StorageBackend, Dict[str, Dataset]):
        inference_backend: InferenceBackend = arguments.inference_backend()
        inference_backend.set_verbosity(INFERENCE_BACKEND_VERBOSE)
        storage_backend = STORAGE_BACKEND_REGISTRY.get(STORAGE_BACKEND)()
        cached_datasets: Dict[str, Dataset] = {}

        return inference_backend, storage_backend, cached_datasets

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
