from abc import ABC, abstractmethod
from typing import TypedDict, Dict, List

from dataset import SingleDataInstance
from decoder import GreedyConstrainedDecoder, GreedyDecoder, TemperatureDecoder, Decoder
from completion import CompletionResult, CompletionHistory, CompletionConfig
from prompt import PromptChain, PromptCompletionStep


class SingleBenchmarkResult(TypedDict):
    question_id: int
    question_id_string: str
    question: str
    answers: List[str]
    labels: List[str]
    model_choice: str
    correct_answer: str
    is_correct: bool
    last_prompt: str
    completions: List[CompletionHistory]


class Metrics(TypedDict):
    total_results: int
    num_correct: int
    accuracy: float
    total_prompt_tokens: int
    total_prompt_eval_ms: float
    total_generated_tokens: int
    total_generation_ms: float


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

    @staticmethod
    def get_default() -> "Benchmark":
        return NonCoTStandardBenchmark()

    @staticmethod
    def get_by_name(benchmark_name: str) -> "Benchmark":
        return benchmark_mapping[benchmark_name]()

    @abstractmethod
    def prompt_chains(self, single_data_instance: SingleDataInstance) -> List[PromptChain]:
        raise NotImplementedError

    @abstractmethod
    def compute_single_result(self, single_data_instance: SingleDataInstance, prompt_chain_results: List[CompletionHistory]) -> SingleBenchmarkResult:
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, all_results: List[SingleBenchmarkResult]) -> Dict[str, float | int]:
        raise NotImplementedError


class LabelGenerationBenchmarkType(Benchmark, ABC):
    @staticmethod
    def is_equal(answer: str, reference: str) -> bool:
        return answer.strip() == reference

    @staticmethod
    def default_label_completion_step(single_data_instance: SingleDataInstance) -> PromptCompletionStep:
        return PromptCompletionStep(
            name="label",
            completion_config=CompletionConfig(max_logprobs=50),
            decoder=GreedyConstrainedDecoder(single_data_instance.answer_labels)
        )

    def compute_single_result(self, single_data_instance: SingleDataInstance,
                              prompt_chain_results: List[CompletionHistory]) -> SingleBenchmarkResult:
        answer_label_completion = prompt_chain_results[0].completions["label"].completion_result
        model_selection = answer_label_completion.get_text().strip()
        correct_answer_label = single_data_instance.answer_labels[single_data_instance.correct_key]

        return SingleBenchmarkResult(
            question_id=single_data_instance.id,
            question_id_string=single_data_instance.row_name,
            question=single_data_instance.question,
            answers=single_data_instance.answer_texts,
            labels=single_data_instance.answer_labels,
            correct_answer=correct_answer_label,
            model_choice=model_selection,
            is_correct=self.is_equal(model_selection, correct_answer_label),
            last_prompt=answer_label_completion.prompt,
            completions=prompt_chain_results
        )

    def compute_metrics(self, all_results: List[SingleBenchmarkResult]) -> Dict:
        total_results = len(all_results)
        num_correct = len(list(filter(lambda item: item["is_correct"], all_results)))
        accuracy = round(num_correct * 100 / total_results, 2)

        total_prompt_tokens = 0
        total_prompt_eval_ms = 0
        total_generated_tokens = 0
        total_generation_ms = 0
        for result in all_results:
            for completion in result["completions"][0].completions.values():
                total_prompt_tokens += completion.completion_result.usage.prompt_tokens
                total_prompt_eval_ms += completion.completion_result.usage.prompt_ms
                total_generated_tokens += completion.completion_result.usage.completion_tokens
                total_generation_ms += completion.completion_result.usage.completion_ms

        return Metrics(
            total_results=total_results,
            num_correct=num_correct,
            accuracy=accuracy,
            total_prompt_tokens=total_prompt_tokens,
            total_prompt_eval_ms=round(total_prompt_eval_ms, 2),
            total_generated_tokens=total_generated_tokens,
            total_generation_ms=round(total_generation_ms, 2)
        )


class NonCoTBenchmark(LabelGenerationBenchmarkType, ABC):
    @abstractmethod
    def get_label_prompt(self) -> str:
        raise NotImplementedError

    def get_label_completion_step(self, single_data_instance: SingleDataInstance) -> PromptCompletionStep:
        return self.default_label_completion_step(single_data_instance)

    def prompt_chains(self, single_data_instance: SingleDataInstance) -> List[PromptChain]:
        prompt_chains = [
            PromptChain().add_template(self.default_optional_context_template)
                         .add_template(self.default_question_template)
                         .add_template(self.default_answer_option_template)
                         .add_template(self.get_label_prompt())
                         .add_completion_step(self.get_label_completion_step(single_data_instance))
        ]

        return prompt_chains


class NonCoTStandardBenchmark(NonCoTBenchmark):
    def get_label_prompt(self) -> str:
        return "Among {{ single_data_instance.answer_labels[0] }} through " \
               "{{ single_data_instance.answer_labels[-1] }}, the correct answer is: "


class NonCoTExplicitInstructionBenchmark(NonCoTStandardBenchmark):
    def get_label_prompt(self) -> str:
        return "Among {{ single_data_instance.answer_labels[0] }} through " \
               "{{ single_data_instance.answer_labels[-1] }}, what is the correct answer? \n\n" \
               "Just answer with the correct label, e.g. with {{ single_data_instance.answer_labels[0]}} " \
               "if answer {{ single_data_instance.answer_labels[0] }} is correct."


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


class CoTStandardBenchmark(CoTBenchmark):
    def get_prompt_parts(self) -> CoTPromptParts:
        return CoTPromptParts(
            reasoning_prompt_template="Among {{ single_data_instance.answer_labels[0] }} through "
                                      "{{ single_data_instance.answer_labels[-1] }}, what is the correct answer?\n\n"
                                      "Let's think step by step.",
            label_prompt_template="Given this reasoning, the correct answer is: "
        )


class CoTVariant1Benchmark(CoTStandardBenchmark):
    def get_prompt_parts(self) -> CoTPromptParts:
        prompt_parts = CoTStandardBenchmark.get_prompt_parts(self)
        prompt_parts.label_prompt_template = "Given this reasoning, the correct answer among labels "\
                                             "{{ single_data_instance.answer_labels[0] }} through "\
                                             "{{ single_data_instance.answer_labels[-1] }} is: \n\n"
        return prompt_parts


class CoTVariant2Benchmark(CoTVariant1Benchmark):
    def get_reasoning_completion_step(self) -> PromptCompletionStep:
        return PromptCompletionStep(
            name="reasoning",
            completion_config=CompletionConfig(max_tokens=2048, max_logprobs=1),
            decoder=GreedyDecoder(),
            prefix="<reasoning>\n",
            suffix="\n</reasoning>\n"
        )


class CoTVariant1TemperatureBenchmark(CoTVariant1Benchmark):
    def get_reasoning_completion_step(self) -> PromptCompletionStep:
        return PromptCompletionStep(
            name="reasoning",
            completion_config=CompletionConfig(max_tokens=2048, max_logprobs=1),
            decoder=TemperatureDecoder(temperature=0.8),
            prefix="Reasoning: "
        )


class ScoringBenchmarkType(Benchmark, ABC):
    def compute_single_result(self, single_data_instance: SingleDataInstance, prompt_chain_results: List[List[CompletionResult]]):
        # get highest score
        pass

    def compute_metrics(self, all_results: List[SingleBenchmarkResult]) -> Dict[str, float | int]:
        pass


class NonCoTScoreIndividuallyBenchmark(ScoringBenchmarkType):
    def prompt_chains(self, single_data_instance: SingleDataInstance) -> List[PromptChain]:
        prompt_chains = []

        for text in single_data_instance.answer_texts:
            prompt_chains.append(
                PromptChain().add_template(self.default_optional_context_template)
                             .add_template(self.default_question_template)
                             .add_text(f"Answer: {text}")
                             .get_completion()
            )

        return prompt_chains

benchmark_mapping: Dict[str, callable] = {
    "default": NonCoTStandardBenchmark,
    "non-cot-standard": NonCoTStandardBenchmark,
    "non-cot-instruct": NonCoTExplicitInstructionBenchmark,
    "non-cot-score-individually": NonCoTScoreIndividuallyBenchmark,
    "cot-standard": CoTStandardBenchmark,
    "cot-variant-0": CoTStandardBenchmark,
    "cot-variant-1": CoTVariant1Benchmark,
    "cot-variant-2": CoTVariant2Benchmark,
    "cot-variant-1-temperature": CoTVariant1TemperatureBenchmark,
}
