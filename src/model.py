from abc import ABC
from typing import List, Dict
from jinja2 import Template, FileSystemLoader, Environment


class ModelConfig(ABC):
    model_name: str
    chat_template_name: str
    chat_template: Template
    optional: bool = False

    @staticmethod
    def load_template(template_name: str) -> Template:
        template_loader = FileSystemLoader(searchpath="./chat_templates")
        template_env = Environment(loader=template_loader, trim_blocks=True, lstrip_blocks=True)
        template_file = f"{template_name}.jinja"

        return template_env.get_template(template_file)

    @staticmethod
    def get_all_names():
        return model_mapping.keys()

    @staticmethod
    def get_all_required_names():
        return {k: v for k, v in model_mapping.items() if v.optional is not True}.keys()

    @staticmethod
    def get_by_name(model_name: str):
        return model_mapping[model_name]


class HFModelConfig(ModelConfig):
    hf_repo: str = ""
    hf_filename: str = ""
    hf_tokenizer: str = ""
    use_hf_tokenizer: bool

    def __init__(self, model_name: str, chat_template_name: str, hf_repo: str, hf_tokenizer: str, use_hf_tokenizer: bool, hf_filename: str = "", optional: bool = False):
        self.model_name = model_name
        self.chat_template_name = chat_template_name
        self.chat_template = self.load_template(chat_template_name)
        self.hf_repo = hf_repo
        if hf_filename:
            self.hf_filename = hf_filename
        else:
            self.hf_filename = self.model_name
        self.hf_tokenizer = hf_tokenizer
        self.use_hf_tokenizer = use_hf_tokenizer
        self.optional = optional


model_mapping: Dict[str, ModelConfig] = {
    "llama-2-7b-chat": HFModelConfig(
        model_name="llama-2-7b-chat",
        chat_template_name="llama-2-chat",
        hf_repo="TheBloke/Llama-2-7B-Chat-GGUF",
        hf_tokenizer="meta-llama/Llama-2-7b-chat-hf",
        use_hf_tokenizer=True
    ),
    "llama-2-13b-chat": HFModelConfig(
        model_name="llama-2-13b-chat",
        chat_template_name="llama-2-chat",
        hf_repo="TheBloke/Llama-2-13B-Chat-GGUF",
        hf_tokenizer="meta-llama/Llama-2-13b-chat-hf",
        use_hf_tokenizer=True
    ),
    "orca-2-7b": HFModelConfig(
        model_name="orca-2-7b",
        chat_template_name="orca-2",
        hf_repo="TheBloke/Orca-2-7B-GGUF",
        hf_tokenizer="microsoft/Orca-2-7b",
        use_hf_tokenizer=True
    ),
    "orca-2-13b": HFModelConfig(
        model_name="orca-2-13b",
        chat_template_name="orca-2",
        hf_repo="TheBloke/Orca-2-13B-GGUF",
        hf_tokenizer="microsoft/Orca-2-7b",
        use_hf_tokenizer=True
    ),
    "phi-2": HFModelConfig(
        model_name="phi-2",
        chat_template_name="phi-2",
        hf_repo="TheBloke/Phi-2-GGUF",
        hf_tokenizer="microsoft/phi-2",
        use_hf_tokenizer=True
    ),
    "mistral-7b-instruct-v0.2": HFModelConfig(
        model_name="mistral-7b-instruct-v0.2",
        chat_template_name="mistral-7b-instruct",
        hf_repo="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        hf_tokenizer="mistralai/Mistral-7B-Instruct-v0.2",
        use_hf_tokenizer=True
    ),
    "llama-3-8b": HFModelConfig(
        model_name="llama-3-8b",
        chat_template_name="fallback",
        hf_repo="QuantFactory/Meta-Llama-3-8B-GGUF",
        hf_filename="Meta-Llama-3-8B",
        hf_tokenizer="meta-llama/Meta-Llama-3-8B",
        use_hf_tokenizer=True,
        optional=True
    ),
    "llama-3-8b-instruct": HFModelConfig(
        model_name="llama-3-8b-instruct",
        chat_template_name="llama-3-instruct",
        hf_repo="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        hf_filename="Meta-Llama-3-8B-Instruct",
        hf_tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
        use_hf_tokenizer=True,
        optional=True
    ),
    "gemma-7b-it": HFModelConfig(
        model_name="gemma-7b-it",
        chat_template_name="gemma",
        hf_repo="mlabonne/gemma-7b-it-GGUF",
        hf_tokenizer="google/gemma-7b-it",
        use_hf_tokenizer=True,
        optional=True
    )
}