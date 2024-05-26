from abc import ABC, abstractmethod
from typing import List

import json


class Decoder(ABC):
    temperature: float = 0.0
    repeat_penalty: float = 1.0
    repeat_last_n: int = 64
    min_p: float = 0.0
    top_p: float = 1.0
    top_k: int = 100

    def to_dict(self):
        return {'decoder': self.__class__.__name__, 'attributes': self.__dict__}

    def __repr__(self):
        return json.dumps(self.to_dict())


class GreedyDecoder(Decoder):
    pass


class GreedyConstrainedDecoder(GreedyDecoder):
    allowed_string: List[str]

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


class BeamSearch(Decoder):
    num_beams: int

    def __init__(self, num_beams: int):
        self.num_beams = num_beams
