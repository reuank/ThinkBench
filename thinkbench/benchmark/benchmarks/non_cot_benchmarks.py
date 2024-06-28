from abc import abstractmethod, ABC
from typing import List

from benchmark.benchmark import BENCHMARK_REGISTRY
from benchmark.benchmark_types import LabelGenerationBenchmarkType, ScoringBenchmarkType
from benchmark.prompt_chain import PromptCompletionStep, PromptChain
from constants import DEFAULT_OPTIONAL_CONTEXT_TEMPLATE, DEFAULT_QUESTION_TEMPLATE, DEFAULT_ANSWER_OPTION_TEMPLATE
from dataset.single_data_instance import SingleDataInstance


class NonCoTBenchmark(LabelGenerationBenchmarkType, ABC):
    def __init__(self):
        self.DEFAULT_OPTIONAL_CONTEXT_TEMPLATE = None

    @abstractmethod
    def get_label_prompt(self) -> str:
        raise NotImplementedError

    def get_label_completion_step(self, single_data_instance: SingleDataInstance) -> PromptCompletionStep:
        return self.default_label_completion_step(single_data_instance)

    def prompt_chains(self, single_data_instance: SingleDataInstance) -> List[PromptChain]:
        prompt_chains = [
            PromptChain().add_template(DEFAULT_OPTIONAL_CONTEXT_TEMPLATE)
                         .add_template(DEFAULT_QUESTION_TEMPLATE)
                         .add_template(DEFAULT_ANSWER_OPTION_TEMPLATE)
                         .add_template(self.get_label_prompt())
                         .add_completion_step(self.get_label_completion_step(single_data_instance))
        ]

        return prompt_chains


@BENCHMARK_REGISTRY.register(name="non-cot-standard", is_default=True)
class NonCoTStandardBenchmark(NonCoTBenchmark):
    def get_label_prompt(self) -> str:
        return "Among {{ single_data_instance.answer_labels[0] }} through " \
               "{{ single_data_instance.answer_labels[-1] }}, the correct answer is: "


@BENCHMARK_REGISTRY.register(name="non-cot-standard-bracket")
class NonCoTStandardBracketBenchmark(NonCoTBenchmark):
    def get_label_prompt(self) -> str:
        return "Among {{ single_data_instance.answer_labels[0] }} through " \
               "{{ single_data_instance.answer_labels[-1] }}, the correct answer is: \n("


@BENCHMARK_REGISTRY.register(name="non-cot-variant-1")
class NonCoTVariant1Benchmark(NonCoTBenchmark):
    def get_label_prompt(self) -> str:
        return "Among {{ single_data_instance.answer_labels[0] }} through " \
               "{{ single_data_instance.answer_labels[-1] }}, the correct answer is: \n\n"


@BENCHMARK_REGISTRY.register("non-cot-explicit")
class NonCoTExplicitInstructionBenchmark(NonCoTStandardBenchmark):
    def get_label_prompt(self) -> str:
        return "Just answer with the correct label, e.g. with {{ single_data_instance.answer_labels[0]}} " \
               "if answer {{ single_data_instance.answer_labels[0] }} is correct.\n" \
               "Among {{ single_data_instance.answer_labels[0] }} through " \
               "{{ single_data_instance.answer_labels[-1] }}, what is the correct answer? \n\n"


class NonCoTScoreIndividuallyBenchmark(ScoringBenchmarkType):
    def prompt_chains(self, single_data_instance: SingleDataInstance) -> List[PromptChain]:
        prompt_chains = []

        for text in single_data_instance.answer_texts:
            prompt_chains.append(
                PromptChain().add_template(DEFAULT_OPTIONAL_CONTEXT_TEMPLATE)
                             .add_template(DEFAULT_QUESTION_TEMPLATE)
                             .add_text(f"Answer: {text}")
                             .get_completion()
            )

        return prompt_chains
