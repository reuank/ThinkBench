from typing import List, Dict, Union

import fire

from benchmark.testcase import TestCase
from benchmark.results import TestCaseResult
from dataset.dataset import Dataset
from dataset.single_data_instance import Numbering, Permutation
from inference.backends.llama_cpp_server_backend import LlamaCppServerInferenceBackend
from inference.inference_backend import InferenceBackend
from utils.list_utils import ensure_list
from utils.logger import Logger
from utils.timer import Timer
from storage.storage_backend import StorageBackend


class ThinkBenchArguments:
    def __init__(
        self,
        models: str = "default",
        datasets: str = "default",
        inference_backend: str = "default",
        benchmarks: str = "default",
        storage: str = "default",
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
        self.storage_backend = STORAGE_BACKEND_REGISTRY.get_single(storage)
        self.label_numbering = Numbering(labels)
        self.label_permutation = Permutation(permutation)

        self.limit = limit
        self.n_random = random
        self.use_chat_template = use_chat_template
        self.comment = comment


class ThinkBench:
    @staticmethod
    def run_thinkbench(arguments: ThinkBenchArguments) -> None:
        inference_backend: Union[InferenceBackend, None] = None
        test_case_results: List[TestCaseResult] = []

        try:
            Timer.get_instance("Run all").start_over()

            inference_backend: InferenceBackend = arguments.inference_backend()
            storage_backend: StorageBackend = arguments.storage_backend()

            test_case_results = ThinkBench.run_test_cases(arguments, inference_backend, storage_backend)
        except Exception as e:
            Logger.error(f"An error occurred during the benchmarking process: {e}")
        finally:
            Timer.get_instance("Run all").end(print_timer=True)
            if test_case_results:
                Logger.print_results_table(test_case_results)
            if inference_backend and isinstance(inference_backend, LlamaCppServerInferenceBackend):
                inference_backend.terminate_all_running_servers()

    @staticmethod
    def run_test_cases(
            arguments: ThinkBenchArguments,
            inference_backend: InferenceBackend,
            storage_backend: StorageBackend
    ) -> List[TestCaseResult]:
        test_case_results: List[TestCaseResult] = []

        cached_datasets: Dict[str, Dataset] = {}

        for model_config in arguments.model_configs:
            Logger.print_header(f"Benchmarking model {model_config.model_name}")
            inference_backend.load_model_from_config(model_config)

            for dataset in arguments.datasets:
                if dataset.name not in cached_datasets.keys():
                    cached_datasets[dataset.name] = dataset()
                else:
                    Logger.info(f"Dataset {dataset.name} was already loaded previously. Using cached version.")

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

                    storage_backend.store(test_case_result)
                    test_case_results.append(test_case_result)

        return test_case_results


if __name__ == '__main__':
    from benchmark.benchmark import BENCHMARK_REGISTRY
    from dataset.dataset import DATASET_REGISTRY
    from inference.inference_backend import INFERENCE_BACKEND_REGISTRY
    from model_config.model_config import MODEL_CONFIG_REGISTRY
    from storage.storage_backend import STORAGE_BACKEND_REGISTRY

    def run_thinkbench_cli(**kwargs):
        arguments = ThinkBenchArguments(**kwargs)
        ThinkBench.run_thinkbench(arguments)

    fire.Fire(run_thinkbench_cli)
