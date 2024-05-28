from abc import ABC
from typing import List, Dict

from constants import LABEL_MAX_LOGPROBS
from .benchmark import Benchmark
from benchmark.benchmark import SingleBenchmarkResult
from benchmark.prompt_chain import PromptCompletionStep
from benchmark.results import Metrics
from dataset.single_data_instance import SingleDataInstance
from inference.completion import CompletionConfig, CompletionHistory, CompletionResult
from inference.decoder import GreedyConstrainedDecoder


class LabelGenerationBenchmarkType(Benchmark, ABC):
    @staticmethod
    def is_equal(answer: str, reference: str) -> bool:
        return answer.strip() == reference

    @staticmethod
    def default_label_completion_step(single_data_instance: SingleDataInstance) -> PromptCompletionStep:
        return PromptCompletionStep(
            name="label",
            completion_config=CompletionConfig(max_logprobs=LABEL_MAX_LOGPROBS),
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


class ScoringBenchmarkType(Benchmark, ABC):
    def compute_single_result(self, single_data_instance: SingleDataInstance, prompt_chain_results: List[List[CompletionResult]]):
        # get highest score
        pass

    def compute_metrics(self, all_results: List[SingleBenchmarkResult]) -> Dict[str, float | int]:
        pass
