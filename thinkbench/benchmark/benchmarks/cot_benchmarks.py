from abc import abstractmethod, ABC
from typing import List

from benchmark.benchmark import BENCHMARK_REGISTRY
from benchmark.benchmark_types import LabelGenerationBenchmarkType
from benchmark.prompt_chain import PromptCompletionStep, PromptChain
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
            completion_config=CompletionConfig(max_tokens=2048, max_logprobs=1),
            decoder=GreedyDecoder(),
            prefix="Reasoning: "
        )

    def prompt_chains(self, single_data_instance: SingleDataInstance) -> List[PromptChain]:
        reasoning_prompt_parts = self.get_prompt_parts()

        prompt_chains = [
            PromptChain().add_template(self.default_optional_context_template)
                         .add_template(self.default_question_template)
                         .add_template(self.default_answer_option_template)
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
        prompt_parts = CoTStandardBenchmark.get_prompt_parts(self)
        prompt_parts.label_prompt_template = "Given this reasoning, the correct answer among labels "\
                                             "{{ single_data_instance.answer_labels[0] }} through "\
                                             "{{ single_data_instance.answer_labels[-1] }} is: \n\n"
        return prompt_parts


@BENCHMARK_REGISTRY.register("cot-variant-1-temperature")
class CoTVariant1TemperatureBenchmark(CoTVariant1Benchmark):
    def get_reasoning_completion_step(self) -> PromptCompletionStep:
        return PromptCompletionStep(
            name="reasoning",
            completion_config=CompletionConfig(max_tokens=2048, max_logprobs=1),
            decoder=TemperatureDecoder(temperature=0.8),
            prefix="Reasoning: "
        )


@BENCHMARK_REGISTRY.register("cot-variant-2")
class CoTVariant2Benchmark(CoTVariant1Benchmark):
    def get_reasoning_completion_step(self) -> PromptCompletionStep:
        return PromptCompletionStep(
            name="reasoning",
            completion_config=CompletionConfig(max_tokens=2048, max_logprobs=1),
            decoder=GreedyDecoder(),
            prefix="<reasoning>\n",
            suffix="\n</reasoning>\n"
        )
