from abc import ABC
from typing import List, Dict
from jinja2 import Template, FileSystemLoader, Environment


class ModelConfig(ABC):
    model_name: str
    chat_template: Template
    bos_token: str = ""
    eos_token: str = ""

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
    def get_by_name(model_name: str):
        return model_mapping[model_name]


class HFModelConfig(ModelConfig):
    hf_repo: str = ""
    hf_tokenizer: str = ""
    use_hf_tokenizer: bool

    def __init__(self, model_name: str, chat_template: Template, hf_repo: str, hf_tokenizer: str, use_hf_tokenizer: bool, bos_token: str, eos_token: str):
        self.model_name = model_name
        self.chat_template = chat_template
        self.hf_repo = hf_repo
        self.hf_tokenizer = hf_tokenizer
        self.use_hf_tokenizer = use_hf_tokenizer
        self.bos_token = bos_token
        self.eos_token = eos_token


model_mapping: Dict[str, ModelConfig] = {
    "llama-2-7b-chat": HFModelConfig(
        model_name="llama-2-7b-chat",
        chat_template=ModelConfig.load_template("llama-2-7b-chat"),
        hf_repo="TheBloke/Llama-2-7B-Chat-GGUF",
        hf_tokenizer="meta-llama/Llama-2-7b-chat-hf",
        use_hf_tokenizer=True,
        bos_token="<s>",
        eos_token="</s>"
    ),
    "llama-2-13b-chat": HFModelConfig(
        model_name="llama-2-13b-chat",
        chat_template=ModelConfig.load_template("llama-2-7b-chat"),
        hf_repo="TheBloke/Llama-2-13B-Chat-GGUF",
        hf_tokenizer="meta-llama/Llama-2-13b-chat-hf",
        use_hf_tokenizer=True,
        bos_token="<s>",
        eos_token="</s>"
    ),
    "orca-2-7b": HFModelConfig(
        model_name="orca-2-7b",
        chat_template=ModelConfig.load_template("llama-2-7b-chat"),
        hf_repo="TheBloke/Orca-2-7B-GGUF",
        hf_tokenizer="microsoft/Orca-2-7b",
        use_hf_tokenizer=True,
        bos_token="<s>",
        eos_token="</s>"
    ),
    "orca-2-13b": HFModelConfig(
        model_name="orca-2-13b",
        chat_template=ModelConfig.load_template("llama-2-7b-chat"),
        hf_repo="TheBloke/Orca-2-13B-GGUF",
        hf_tokenizer="microsoft/Orca-2-7b",
        use_hf_tokenizer=True,
        bos_token="<s>",
        eos_token="</s>"
    )
}