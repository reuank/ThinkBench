import json
from typing import List, Dict

from numpy import float32


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float32):
            return float(obj)
        else:
            return obj.__dict__


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
    completion_tokens: int
    total_tokens: int

    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
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