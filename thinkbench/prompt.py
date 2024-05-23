from abc import ABC, abstractmethod
from jinja2 import Template

from completion import CompletionConfig
from decoder import Decoder


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
        self.template = Template(template_string)

    def __repr__(self):
        return {"step_type": "template", "template_string": self.template_string}


class PromptCompletionStep(PromptStep):
    name: str
    max_tokens: int
    max_logprobs: int
    decoder: Decoder
    prefix: str
    suffix: str

    def __init__(self, name: str, max_tokens: int = 1, max_logprobs: int = 10, decoder: Decoder = None, prefix: str = "", suffix: str = ""):
        self.name = name
        self.completion_config = CompletionConfig(max_tokens=max_tokens, max_logprobs=max_logprobs)
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

    def get_completion(self, name: str, max_tokens: int = 1, max_logprobs: int = 10, decoder: Decoder = None, prefix: str = "", suffix: str = "") -> "PromptChain":
        self.steps.append(
            PromptCompletionStep(
                name=name,
                max_tokens=max_tokens,
                max_logprobs=max_logprobs,
                decoder=decoder,
                prefix=prefix,
                suffix=suffix
            )
        )
        return self

    def __repr__(self):
        return str([step.__repr__() for step in self.steps])
