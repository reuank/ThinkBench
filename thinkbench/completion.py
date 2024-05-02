import json
from typing import List, Dict

from numpy import float32

from decoder import Decoder, GreedyConstrainedDecoder


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float32):
            return float(obj)
        else:
            return obj.__dict__


class CompletionConfig:
    max_tokens: int
    temperature: float
    max_logprobs: int
    echo: bool
    repeat_penalty: float  # penalty of 1.0 fixes selection of less likely yet valid token when using grammars

    def __init__(self, max_tokens: int = 1, temperature: float = 0.0, max_logprobs: int = 10, echo: bool = False, repeat_penalty: float = 1.0):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_logprobs = max_logprobs
        self.echo = echo
        self.repeat_penalty = repeat_penalty

    def __repr__(self):
        return json.dumps(self, cls=NumpyEncoder, indent=2)


class Logprobs:
    tokens: List[str]
    text_offset: List[int]
    token_logprobs: List[float]
    top_logprobs: List[Dict[str, float]]

    def __init__(self, tokens: List[str], text_offset: List[int], token_logprobs: List[float], top_logprobs: List[Dict[str, float]]) -> None:
        self.tokens = tokens
        self.text_offset = text_offset
        self.token_logprobs = token_logprobs
        self.top_logprobs = top_logprobs

    def __repr__(self):
        return json.dumps(self, indent=2, cls=NumpyEncoder)


class Choice:
    text: str
    index: int
    logprobs: Logprobs
    finish_reason: str

    def __init__(self, text: str, index: int, logprobs: Logprobs, finish_reason: str) -> None:
        self.text = text
        self.index = index
        self.logprobs = logprobs
        self.finish_reason = finish_reason

    def __repr__(self):
        return json.dumps(self, indent=2, cls=NumpyEncoder)


class Usage:
    prompt_tokens: int
    prompt_tokens_per_second: float
    prompt_ms: float
    completion_tokens: int
    completion_tokens_per_second: float
    completion_ms: float
    total_tokens: int

    def __init__(self, prompt_tokens: int, prompt_tokens_per_second: float, prompt_ms: float, completion_tokens: int,
                 completion_tokens_per_second: float, completion_ms: float, total_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.prompt_tokens_per_second = prompt_tokens_per_second
        self.prompt_ms = prompt_ms

        self.completion_tokens = completion_tokens
        self.completion_tokens_per_second = completion_tokens_per_second
        self.completion_ms = completion_ms

        self.total_tokens = total_tokens

    def __repr__(self):
        return json.dumps(self, indent=2, cls=NumpyEncoder)


class CompletionResult:
    prompt: str
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

    def __init__(self, prompt: str, id: str, object: str, created: int, model: str, choices: List[Choice], usage: Usage) -> None:
        self.prompt = prompt
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage

    def __repr__(self):
        return json.dumps(self, indent=2, cls=NumpyEncoder)

    def get_text(self):
        return self.choices[0].text

    def get_usage(self):
        return self.usage

    def get_most_probable_token(self):
        return self.choices[0].logprobs.tokens[0]


class FinishedCompletion:
    completion_result: CompletionResult
    decoder: Decoder
    completion_config: CompletionConfig

    def __init__(self, completion_result: CompletionResult, decoder: Decoder, completion_config: CompletionConfig):
        self.completion_result = completion_result
        self.decoder = decoder
        self.completion_config = completion_config


class CompletionHistory:
    completions: Dict[str, FinishedCompletion]

    def __init__(self):
        self.completions = {}

    def add_completion(self, name: str, completion_result: CompletionResult, decoder: Decoder, completion_config: CompletionConfig):
        self.completions[name] = FinishedCompletion(completion_result, decoder, completion_config)

    def get_texts(self) -> Dict[str, str]:
        texts: Dict[str, str] = {}

        for k, v in self.completions.items():
            texts[k] = v.completion_result.get_text()

        return texts

    def to_dict(self) -> Dict[str, Dict[str, str | Dict]]:
        completions_dict: Dict[str, Dict[str, str | Dict]] = {}

        for k, v in self.completions.items():
            completions_dict[k] = {
                "text": v.completion_result.get_text(),
                "stats": v.completion_result.get_usage(),
                "config": v.completion_config,
                "decoder": v.decoder
            }
            if isinstance(v.decoder, GreedyConstrainedDecoder):
                completions_dict[k]["logprobs"] = v.completion_result.choices[0].logprobs.top_logprobs[0]

        return completions_dict

    def __repr__(self):
        return json.dumps(self.to_dict())
