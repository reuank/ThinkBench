from abc import ABC, abstractmethod
from typing import Dict, List

from benchmark.results import SingleBenchmarkResult
from dataset.single_data_instance import SingleDataInstance
from inference.completion import CompletionHistory
from benchmark.prompt_chain import PromptChain
from utils.registry import Registry


class Benchmark(ABC):
    default_optional_context_template = (
        "{% if single_data_instance.context %}"
        "Passage:\n"
        "{{ single_data_instance.context }}"
        "\n\n"
        "{% endif %}"
    )

    default_question_template = (
        "Question:\n"
        "{{ single_data_instance.question }}"
        "\n\n"
    )

    default_answer_option_template = (
        "Answer Choices:\n"
        "{% for label in single_data_instance.answer_labels %}"
        "({{ label }}) {{ single_data_instance.answer_texts[loop.index0] }}{{ '\n' if not loop.last }}"
        "{% endfor %}"
        "\n\n"
    )

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
