from abc import ABC, abstractmethod
from jinja2 import Template

from decoder import CompletionConfig, Decoder


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
    max_tokens: int
    max_logprobs: int
    decoder: Decoder

    def __init__(self, max_tokens: int = 1, max_logprobs: int = 10, decoder: Decoder = None):
        self.completion_config = CompletionConfig(max_tokens=max_tokens, max_logprobs=max_logprobs)
        self.decoder = decoder

    def __repr__(self):
        return {"step_type": "completion", "max_tokens": self.completion_config.max_tokens, "max_logprobs": self.completion_config.max_logprobs, "decoder": self.decoder}


class PromptChain:
    default_question_template = (
        "Question:\n"
        "{{ single_data_instance.question }}"
        "\n\n"
    )

    default_answer_option_template = (
        "Answer Choices:\n"
        "{% for label in single_data_instance.answer_labels %}"
        "({{ label }}) {{ single_data_instance.answer_texts[loop.index0] }}{{ '\n' if not loop.last }}"
        "{% endfor %}"
        "\n\n"
    )

    def __init__(self):
        self.steps: list[PromptStep] = []

    def add_default_question_template(self):
        self.add_template(self.default_question_template)
        return self

    def add_default_answer_options_template(self):
        self.add_template(self.default_answer_option_template)
        return self

    def add_newline(self):
        self.steps.append(PromptTextStep("\n"))
        return self
    
    def add_template(self, template_string: str):
        self.steps.append(PromptTemplateStep(template_string))
        return self

    def add_text(self, context: str):
        self.steps.append(PromptTextStep(context))
        return self

    def get_completion(self, max_tokens: int = 1, max_logprobs: int = 10, decoder: Decoder = None):
        self.steps.append(
            PromptCompletionStep(
                max_tokens=max_tokens,
                max_logprobs=max_logprobs,
                decoder=decoder
            )
        )
        return self

    def __repr__(self):
        return str([step.__repr__() for step in self.steps])
