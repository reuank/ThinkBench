from typing import List, Dict, Union

import fire

from benchmark.testcase import TestCase
from benchmark.results import TestCaseResult
from constants import LIBRARY_ROOT
from dataset.dataset import Dataset
from dataset.single_data_instance import Numbering, Permutation
from inference.backends.llama_cpp_server_backend import LlamaCppServerInferenceBackend
from inference.inference_backend import InferenceBackend
from trace_analysis.classification.classification_evaluator import ClassificationEvaluator
from trace_analysis.classification.automatic_trace_classifier import AutomaticTraceClassifier, TraceClass
from trace_analysis.classification.manual_trace_classifier import ManualTraceClassifier
from trace_analysis.statistics.choice_prob import ChoiceProb
from trace_analysis.statistics.class_accuracy import ClassAccuracy
from trace_analysis.statistics.label_confusion import LabelConfusion
from trace_analysis.statistics.runs_match import RunsMatch
from utils.env_loader import EnvReader
from utils.list_utils import ensure_list
from utils.logger import Logger
from utils.test_case_result_helper import TestCaseResultHelper
from utils.timer import Timer
from storage.storage_backend import StorageBackend


class TestArguments:
    def __init__(
        self,
        models: str,
        datasets: str,
        inference_backend: str,
        benchmarks: str,
        storage: str,
        limit: int,
        random: int,
        labels: str,
        permutation: str,
        use_chat_template: bool,
        comment: str
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
    def run_test(arguments: TestArguments) -> None:
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
            arguments: TestArguments,
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

                    storage_backend.store_test_case_result(test_case_result)
                    test_case_results.append(test_case_result)

        return test_case_results


def run_test_cli(
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
    arguments = TestArguments(
        models, datasets, inference_backend, benchmarks, storage, limit,
        random, labels, permutation, use_chat_template, comment
    )
    ThinkBench.run_test(arguments)


def classify_traces_cli(
    cot_results_path: str,
    non_cot_results_path: str,
    evaluate: bool = True,
    override: bool = False
):
    cot_test_case_results, non_cot_test_case_results = TestCaseResultHelper.load_two_runs(cot_results_path, non_cot_results_path)
    TestCaseResultHelper.ensure_reasoning_present(cot_test_case_results)

    Logger.print_header("Manual Classification")
    manual_classifications = ManualTraceClassifier.classify_test_case_results(
        cot_test_case_results=cot_test_case_results,
        non_cot_test_case_results=non_cot_test_case_results,
        override=override
    )
    ManualTraceClassifier.store_classification_results(manual_classifications)

    Logger.print_header("Automatic Classification")
    complete_classifications = AutomaticTraceClassifier.classify_test_case_results(
        cot_test_case_results=cot_test_case_results,
        non_cot_test_case_results=non_cot_test_case_results,
        override=override
    )
    AutomaticTraceClassifier.store_classification_results(complete_classifications)

    if evaluate:
        Logger.print_header("Classification Evaluation")
        ClassificationEvaluator.evaluate_classifications(complete_classifications)


def analyze_cli(
        cot_results_path: str,
        non_cot_results_path: str,
        class_id: int = -1,

        # Analysis types
        class_accuracy: bool = False,

        runs_match: bool = False,

        label_confusion: bool = False,
        ignore_label_edge_cases: bool = True,
        analyze_run: str = "cot",

        choice_prob: bool = False
):
    if class_id != -1 and class_id not in TraceClass.get_ids():
        Logger.error(f"Class ID {class_id} does not exist. Computing stats for all categories instead.")
        class_id = -1

    cot_test_case_results, non_cot_test_case_results = TestCaseResultHelper.load_two_runs(cot_results_path, non_cot_results_path)
    TestCaseResultHelper.ensure_reasoning_present(cot_test_case_results)

    if class_accuracy:
        ClassAccuracy.compute_all(
            cot_test_case_results=cot_test_case_results,
            non_cot_test_case_results=non_cot_test_case_results,
            class_id=class_id
        )

    if runs_match:
        RunsMatch.compute_all(
            cot_test_case_results=cot_test_case_results,
            non_cot_test_case_results=non_cot_test_case_results,
            class_id=class_id
        )

    if label_confusion:
        LabelConfusion.compute_all(
            cot_test_case_results=cot_test_case_results,
            non_cot_test_case_results=non_cot_test_case_results,
            ignore_label_edge_cases=ignore_label_edge_cases,
            analyze_run=analyze_run
        )

    if choice_prob:
        ChoiceProb.compute_all(
            cot_test_case_results=cot_test_case_results,
            non_cot_test_case_results=non_cot_test_case_results,
            class_id=class_id
        )

    if not class_accuracy and not runs_match and not label_confusion and not choice_prob:
        Logger.info("Please select a metric to compute.")


if __name__ == '__main__':
    from benchmark.benchmark import BENCHMARK_REGISTRY
    from dataset.dataset import DATASET_REGISTRY
    from inference.inference_backend import INFERENCE_BACKEND_REGISTRY
    from model_config.model_config import MODEL_CONFIG_REGISTRY
    from storage.storage_backend import STORAGE_BACKEND_REGISTRY

    EnvReader.load_env_file(f"{LIBRARY_ROOT}/.env")

    fire.Fire({
        'run-test': run_test_cli,
        'classify-traces': classify_traces_cli,
        'analyze': analyze_cli,
    })
