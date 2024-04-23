from abc import ABC
from typing import List

import json


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
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)


class Decoder(ABC):
    def __repr__(self):
        obj_representation = {'decoder_type': self.__class__.__name__, 'attributes': self.__dict__}
        return json.dumps(obj_representation)


class GreedyDecoder(Decoder):
    pass


class GreedyConstrainedDecoder(GreedyDecoder):
    allowed_string: List[str]

    def __init__(self, allowed_strings: List[str]):
        self.allowed_strings = allowed_strings


class BeamSearch(Decoder):
    num_beams: int

    def __init__(self, num_beams: int):
        self.num_beams = num_beams