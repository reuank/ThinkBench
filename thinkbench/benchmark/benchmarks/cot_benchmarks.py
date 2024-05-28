from abc import abstractmethod, ABC
from typing import List

from benchmark.benchmark import BENCHMARK_REGISTRY
from benchmark.benchmark_types import LabelGenerationBenchmarkType
from benchmark.prompt_chain import PromptCompletionStep, PromptChain
from constants import DEFAULT_OPTIONAL_CONTEXT_TEMPLATE, DEFAULT_ANSWER_OPTION_TEMPLATE, DEFAULT_QUESTION_TEMPLATE, \
    REASONING_MAX_TOKENS, REASONING_MAX_LOGPROBS
from dataset.single_data_instance import SingleDataInstance
from inference.completion import CompletionConfig
from inference.decoder import GreedyDecoder, TemperatureDecoder


class CoTPromptParts:
    def __init__(self, reasoning_prompt_template: str, label_prompt_template):
        self.reasoning_prompt_template = reasoning_prompt_template
        self.label_prompt_template = label_prompt_template


class CoTBenchmark(LabelGenerationBenchmarkType, ABC):
    @abstractmethod
    def get_prompt_parts(self) -> CoTPromptParts:
        raise NotImplementedError

    def get_reasoning_completion_step(self) -> PromptCompletionStep:
        return PromptCompletionStep(
            name="reasoning",
            completion_config=CompletionConfig(max_tokens=REASONING_MAX_TOKENS, max_logprobs=REASONING_MAX_LOGPROBS),
            decoder=GreedyDecoder(),
            prefix="Reasoning: "
        )

    def prompt_chains(self, single_data_instance: SingleDataInstance) -> List[PromptChain]:
        reasoning_prompt_parts = self.get_prompt_parts()

        prompt_chains = [
            PromptChain().add_template(DEFAULT_OPTIONAL_CONTEXT_TEMPLATE)
                         .add_template(DEFAULT_QUESTION_TEMPLATE)
                         .add_template(DEFAULT_ANSWER_OPTION_TEMPLATE)
                         .add_template(reasoning_prompt_parts.reasoning_prompt_template)
                         .add_completion_step(self.get_reasoning_completion_step())
                         .add_template(reasoning_prompt_parts.label_prompt_template)
                         .add_completion_step(self.default_label_completion_step(single_data_instance))
        ]

        return prompt_chains


@BENCHMARK_REGISTRY.register("cot-standard")
class CoTStandardBenchmark(CoTBenchmark):
    def get_prompt_parts(self) -> CoTPromptParts:
        return CoTPromptParts(
            reasoning_prompt_template="Among {{ single_data_instance.answer_labels[0] }} through "
                                      "{{ single_data_instance.answer_labels[-1] }}, what is the correct answer?\n\n"
                                      "Let's think step by step.",
            label_prompt_template="Given this reasoning, the correct answer is: "
        )


@BENCHMARK_REGISTRY.register("cot-variant-1")
class CoTVariant1Benchmark(CoTStandardBenchmark):
    def get_prompt_parts(self) -> CoTPromptParts:
        prompt_parts = super().get_prompt_parts()
        prompt_parts.label_prompt_template = "Given this reasoning, the correct answer among labels "\
                                             "{{ single_data_instance.answer_labels[0] }} through "\
                                             "{{ single_data_instance.answer_labels[-1] }} is: \n\n"
        return prompt_parts


@BENCHMARK_REGISTRY.register("cot-variant-1-temperature")
class CoTVariant1TemperatureBenchmark(CoTVariant1Benchmark):
    def get_reasoning_completion_step(self) -> PromptCompletionStep:
        reasoning_completion_step = super().get_reasoning_completion_step()
        reasoning_completion_step.decoder = TemperatureDecoder(temperature=0.8)

        return reasoning_completion_step


@BENCHMARK_REGISTRY.register("cot-variant-1-xml")
class CoTVariant1XMLBenchmark(CoTVariant1Benchmark):
    def get_reasoning_completion_step(self) -> PromptCompletionStep:
        reasoning_completion_step = super().get_reasoning_completion_step()
        reasoning_completion_step.prefix = "<reasoning>\n"
        reasoning_completion_step.suffix = "\n</reasoning>\n"

        return reasoning_completion_step
