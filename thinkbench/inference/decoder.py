from abc import ABC
from typing import List

import json

from constants import *


class Decoder(ABC):
    temperature: float = GREEDY_DECODER_TEMPERATURE
    repeat_penalty: float = DEFAULT_DECODER_REPEAT_PENALTY
    repeat_last_n: int = DEFAULT_DECODER_REPEAT_LAST_N
    min_p: float = DEFAULT_DECODER_MIN_P
    top_p: float = DEFAULT_DECODER_TOP_P
    top_k: int = DEFAULT_DECODER_TOP_K

    def to_dict(self):
        return {'decoder': self.__class__.__name__, 'attributes': self.__dict__}

    def __repr__(self):
        return json.dumps(self.to_dict())


class GreedyDecoder(Decoder):
    pass


class GreedyConstrainedDecoder(GreedyDecoder):
    def __init__(self, allowed_strings: List[str]):
        self.allowed_strings = allowed_strings


class TemperatureDecoder(Decoder):
    def __init__(self, temperature: float):
        self.temperature = temperature


class TopPSamplingDecoder(Decoder):
    def __init__(self, top_p: float, min_p: float):
        self.top_p = top_p
        self.min_p = min_p


class TopKSamplingDecoder(Decoder):
    def __init__(self, top_k: int):
        self.top_k = top_k


class Beam:
    def __init__(self, prompt: str, generated_tokens: List[str], log_prob_sum: float, log_probs: List[float]):
        self.prompt = prompt
        self.generated_tokens = generated_tokens
        self.log_prob_sum = log_prob_sum
        self.log_probs = log_probs

    def __lt__(self, other: "Beam"):
        return self.log_prob_sum < other.log_prob_sum

    def get_current_prompt(self):
        return self.prompt + self.get_completion()

    def get_completion(self):
        return "".join(self.generated_tokens)

    # https://github.com/vllm-project/vllm/blob/290f4ada2bf42174a53ae6aab2873e115c8ae11b/vllm/sequence.py#L335
    def get_beam_search_score(
        self,
        length_penalty: float = 1.0
    ) -> float:
        # return self.log_prob_sum
        return self.log_prob_sum / (len(self.generated_tokens) ** length_penalty)

    def __repr__(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=2)


class BeamSearchDecoder(Decoder):
    def __init__(self, beam_width: int):
        self.beam_width = beam_width
