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


class NucleusDecoder(Decoder):
    def __init__(self, top_p: float):
        self.top_p = top_p


class TopKSamplingDecoder(Decoder):
    def __init__(self, top_k: int):
        self.top_k = top_k


class BeamSearch(Decoder):
    num_beams: int

    def __init__(self, num_beams: int):
        self.num_beams = num_beams
