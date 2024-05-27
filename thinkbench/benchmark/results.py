from typing import TypedDict, Optional, Dict, List

from inference.completion import CompletionHistory


class Metrics(TypedDict):
    total_results: int
    num_correct: int
    accuracy: float
    total_prompt_tokens: int
    total_prompt_eval_ms: float
    total_generated_tokens: int
    total_generation_ms: float


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


class TestCaseResult(TypedDict):
    uuid: str
    model: str
    dataset_name: str
    benchmark_name: str
    label_numbering: str
    label_permutation: str
    n_random: int
    random_seed: int
    hostname: str
    inference_backend: str
    inference_backend_properties: Optional[Dict]
    metrics: Metrics
    start_time: float
    end_time: float
    execution_seconds: float
    current_commit_hash: str
    comment: str
    use_chat_template: bool
    results: List[SingleBenchmarkResult]
