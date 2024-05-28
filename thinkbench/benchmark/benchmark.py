from abc import ABC, abstractmethod
from typing import Dict, List

from benchmark.results import SingleBenchmarkResult
from dataset.single_data_instance import SingleDataInstance
from inference.completion import CompletionHistory
from benchmark.prompt_chain import PromptChain
from utils.registry import Registry


class Benchmark(ABC):
    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def prompt_chains(self, single_data_instance: SingleDataInstance) -> List[PromptChain]:
        raise NotImplementedError

    @abstractmethod
    def compute_single_result(self, single_data_instance: SingleDataInstance, prompt_chain_results: List[CompletionHistory]) -> SingleBenchmarkResult:
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, all_results: List[SingleBenchmarkResult]) -> Dict[str, float | int]:
        raise NotImplementedError


BENCHMARK_REGISTRY = Registry(
    registry_name="benchmarks",
    base_class=Benchmark,
    lazy_load_dirs=["benchmark/benchmarks"]
)
