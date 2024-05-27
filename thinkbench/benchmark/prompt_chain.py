from abc import ABC, abstractmethod
from jinja2 import Template, Environment

from inference.completion import CompletionConfig
from inference.decoder import Decoder


class PromptStep(ABC):
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class PromptTextStep(PromptStep):
    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return {"step_type": "text", "text": self.text}


class PromptTemplateStep(PromptStep):
    template_string: str = ""
    template: Template

    def __init__(self, template_string: str):
        self.template_string = template_string
        self.template = Environment(keep_trailing_newline=True).from_string(template_string)

    def __repr__(self):
        return {"step_type": "template", "template_string": self.template_string}


class PromptCompletionStep(PromptStep):
    name: str
    max_tokens: int
    max_logprobs: int
    decoder: Decoder
    prefix: str
    suffix: str

    def __init__(self, name: str, completion_config: CompletionConfig, decoder: Decoder, prefix: str = "", suffix: str = ""):
        self.name = name
        self.completion_config = completion_config
        self.decoder = decoder
        self.prefix = prefix
        self.suffix = suffix

    def __repr__(self):
        return {"step_type": "completion", "max_tokens": self.completion_config.max_tokens, "max_logprobs": self.completion_config.max_logprobs, "decoder": self.decoder}


class PromptChain:
    def __init__(self):
        self.steps: list[PromptStep] = []

    def add_newline(self) -> "PromptChain":
        self.steps.append(PromptTextStep("\n"))
        return self
    
    def add_template(self, template_string: str) -> "PromptChain":
        self.steps.append(PromptTemplateStep(template_string))
        return self

    def add_text(self, context: str) -> "PromptChain":
        self.steps.append(PromptTextStep(context))
        return self

    def add_completion_step(self, step: PromptStep) -> "PromptChain":
        self.steps.append(step)
        return self

    def get_completion(self, name: str, completion_config: CompletionConfig, decoder: Decoder, prefix: str = "", suffix: str = "") -> "PromptChain":
        self.steps.append(
            PromptCompletionStep(
                name=name,
                completion_config=completion_config,
                decoder=decoder,
                prefix=prefix,
                suffix=suffix
            )
        )
        return self

    def __repr__(self):
        return str([step.__repr__() for step in self.steps])
