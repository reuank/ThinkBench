from abc import ABC, abstractmethod
from typing import List

import json


class Decoder(ABC):
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
    temperature: float

    def __init__(self, temperature: float):
        self.temperature = temperature


class BeamSearch(Decoder):
    num_beams: int

    def __init__(self, num_beams: int):
        self.num_beams = num_beams