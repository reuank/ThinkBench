import json
from collections import deque
from enum import Enum
from typing import List


class Numbering(Enum):
    UNCHANGED = "unchanged"
    LETTERS = "letters"
    NUMBERS = "numbers"
    ROMAN = "roman"

    @staticmethod
    def get_default():
        return Numbering.UNCHANGED


class Permutation(Enum):
    UNCHANGED = "unchanged"
    ADVANCE1 = "advance-1"
    ADVANCE2 = "advance-2"

    @staticmethod
    def get_default():
        return Permutation.UNCHANGED


class SingleDataInstance:
    def __init__(self, id: int, row_name: str, context: str, question: str, answer_texts: List[str], answer_labels: List[str], correct_key: int):
        self.id = id
        self.row_name = row_name
        self.context = context
        self.question = question
        self.answer_texts: List[str] = answer_texts
        self.answer_labels: List[str] = answer_labels
        self.correct_key: int = correct_key

    @staticmethod
    def get_default_labels(count: int):
        if count > 26:
            raise ValueError("Label count exceeds letters in the English alphabet.")
        return [chr(idx + 65) for idx in range(count)]

    @staticmethod
    def get_dummy():
        return SingleDataInstance(
            id=100,
            row_name="row",
            context="{Context}",
            question="{Question}",
            answer_texts=["Text 1", "Text 2"],
            answer_labels=["A", "B"],
            correct_key=0
        )

    def substitute_labels(self, numbering: Numbering):
        if numbering == Numbering.UNCHANGED:
            pass
        elif numbering == Numbering.LETTERS:
            if all(label.isnumeric() for label in self.answer_labels):
                self.answer_labels = [f"{chr(64 + int(x))}" for x in self.answer_labels]  # Generate ["A", "B"] from ["1", "2"]
        elif numbering == Numbering.NUMBERS:
            self.answer_labels = [f"{x + 1}" for x in range(len(self.answer_labels))]  # Generate ["1", "2"] from ["A", "B"]
        else:
            raise ValueError(f"Numbering {numbering.value} not implemented.")

        return self

    @staticmethod
    def rotate_list(list_to_rotate: List, rotations: int):
        rotated_list = deque(list_to_rotate)
        rotated_list.rotate(rotations)

        return list(rotated_list)

    def permute_labels(self, permutation: Permutation):
        correct_answer = self.answer_texts[self.correct_key]

        if permutation == Permutation.UNCHANGED:
            pass
        elif permutation == Permutation.ADVANCE1:
            self.answer_texts = self.rotate_list(self.answer_texts, 1)
            self.correct_key = self.answer_texts.index(correct_answer)
        elif permutation == Permutation.ADVANCE2:
            self.answer_texts = self.rotate_list(self.answer_texts, 2)
            self.correct_key = self.answer_texts.index(correct_answer)
        else:
            raise ValueError(f"Permutation {permutation.value} not implemented.")

        return self

    def __repr__(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)
        # return f"""{self.id=}, {self.row_name=}, {self.context=}, {self.question=}, {self.answer_texts=}, {self.answer_labels=}, {self.correct_key=}"""
