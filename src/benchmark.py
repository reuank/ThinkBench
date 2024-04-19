from abc import ABC, abstractmethod
from typing import TypedDict, Dict, List, Optional, Any

from dataset import SingleDataInstance
from decoder import GreedyConstrainedDecoder, GreedyDecoder
from completion_result import CompletionResult
from prompt import PromptChain


class SingleBenchmarkResult(TypedDict):
    question_id: int
    question_id_string: str
    question: str
    answers: List[str]
    labels: List[str]
    prompt: str
    grammar_string: str
    model_choice: str
    correct_answer: str
    is_correct: bool
    top_logprobs: Dict[str, float]


class Benchmark(ABC):
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
    def compute_single_result(self, single_data_instance: SingleDataInstance, prompt_chain_results: List[List[CompletionResult]]) -> SingleBenchmarkResult:
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, all_results: List[SingleBenchmarkResult]) -> Dict[str, float | int]:
        raise NotImplementedError


class NonCoTStandardBenchmark(Benchmark):
    def prompt_chains(self, single_data_instance: SingleDataInstance) -> List[PromptChain]:
        prompt_chains = [
            PromptChain().add_default_question_template()
                         .add_default_answer_options_template()
                         .add_template("Among {{ single_data_instance.answer_labels[0] }} through "
                                       "{{ single_data_instance.answer_labels[-1] }}, the correct answer is: ")
                         .get_completion(decoder=GreedyConstrainedDecoder(single_data_instance.answer_labels))
        ]

        return prompt_chains

    def compute_single_result(self, single_data_instance: SingleDataInstance, prompt_chain_results: List[List[CompletionResult]]) -> SingleBenchmarkResult:
        relevant_completion = prompt_chain_results[0][0]

        model_selection = relevant_completion.choices[0].logprobs.tokens[0]
        correct_answer_label = single_data_instance.answer_labels[single_data_instance.correct_key]

        #top_logprobs = relevant_completion.choices[0].logprobs.top_logprobs[0]
        #top_logprobs_sorted = sorted(top_logprobs.items(), key=lambda x: x[1], reverse=True)
        #top_logprobs_sorted = dict(top_logprobs_sorted)

        return SingleBenchmarkResult(
            question_id=single_data_instance.id,
            question_id_string=single_data_instance.row_name,
            question=single_data_instance.question,
            answers=single_data_instance.answer_texts,
            labels=single_data_instance.answer_labels,
            prompt=relevant_completion.prompt,
            model_choice=model_selection,
            correct_answer=correct_answer_label,
            is_correct=self.is_equal(model_selection, correct_answer_label),
            top_logprobs=relevant_completion.choices[0].logprobs.top_logprobs[0],
            grammar_string=""
        )

    @staticmethod
    def is_equal(answer: str, reference: str):
        return answer.strip() == reference

    def compute_metrics(self, all_results: List[SingleBenchmarkResult]) -> Dict:
        total_results = len(all_results)
        num_correct = len(list(filter(lambda item: item["is_correct"], all_results)))
        accuracy = round(num_correct / total_results, 2) * 100

        return {
            "total_results": total_results,
            "num_correct": num_correct,
            "accuracy": accuracy
        }


class NonCoTExplicitInstructionBenchmark(NonCoTStandardBenchmark):
    def prompt_chains(self, single_data_instance: SingleDataInstance) -> List[PromptChain]:
        prompt_chains = [
            PromptChain().add_default_question_template()
                .add_default_answer_options_template()
                .add_template("Among {{ single_data_instance.answer_labels[0] }} through "
                              "{{ single_data_instance.answer_labels[-1] }}, what is the correct answer?\n\n")
                .add_template("Just answer with the correct label, e.g. with {{ single_data_instance.answer_labels[0]}}"
                              " if answer {{ single_data_instance.answer_labels[0] }} is correct.")
                .get_completion(decoder=GreedyConstrainedDecoder(single_data_instance.answer_labels))
        ]

        return prompt_chains


class NonCoTScoreIndividuallyBenchmark(Benchmark):
    def prompt_chains(self, single_data_instance: SingleDataInstance) -> List[PromptChain]:
        prompt_chains = []

        for text in single_data_instance.answer_texts:
            prompt_chains.append(
                PromptChain().add_default_question_template()
                             .add_text(f"Answer: {text}")
                             .get_completion()
            )

        return prompt_chains

    def compute_single_result(self, single_data_instance: SingleDataInstance, prompt_chain_results: List[List[CompletionResult]]):
        # get highest score
        pass

    def get_fewshots(self, Data):
        return


class CoTStandardBenchmark(Benchmark):
    def prompt_chains(self, single_data_instance: SingleDataInstance) -> List[PromptChain]:
        prompt_chains = [
            PromptChain()
                .add_default_question_template()
                .add_default_answer_options_template()
                .add_template("Among {{ single_data_instance.answer_labels[0] }} through "
                              "{{ single_data_instance.answer_labels[-1] }}, what is the correct answer?\n\n")
                .add_text("Let's think step by step.")
                .get_completion(max_tokens=1024, decoder=GreedyDecoder(), prefix="Reasoning: ")
                .add_text("Given this reasoning, the correct answer is: ")
                .get_completion(decoder=GreedyConstrainedDecoder(single_data_instance.answer_labels))
        ]

        return prompt_chains

    def compute_single_result(self, single_data_instance: SingleDataInstance,
                              prompt_chain_results: List[List[CompletionResult]]) -> SingleBenchmarkResult:
        pass

    def compute_metrics(self, all_results: List[SingleBenchmarkResult]) -> Dict[str, float | int]:
        pass


benchmark_mapping: Dict[str, callable] = {
    "default": NonCoTStandardBenchmark,
    "non-cot-standard": NonCoTStandardBenchmark,
    "non-cot-instruct": NonCoTExplicitInstructionBenchmark,
    "non-cot-score-individually": NonCoTScoreIndividuallyBenchmark
}
