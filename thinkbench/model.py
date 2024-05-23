import pathlib
from abc import ABC
from enum import Enum
from typing import Dict, Tuple
from jinja2 import Template, FileSystemLoader, Environment


class QuantizationMethod(Enum):
    GGUF = "gguf"
    GPTQ = "gptq"
    AWQ = "awq"


class ModelConfig(ABC):
    model_name: str
    chat_template_name: str
    chat_template: Template
    optional: bool = False

    @staticmethod
    def load_template(template_name: str) -> Template:
        template_loader = FileSystemLoader(searchpath=pathlib.Path(__file__).parent.resolve() / "chat_templates")
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
    def __init__(self,
                 model_name: str,
                 chat_template_name: str,
                 hf_base_model_repo: str,
                 quantized_model_repos: Dict[QuantizationMethod, Tuple[str, str]],
                 use_hf_tokenizer: bool = False,
                 optional: bool = False):
        self.model_name = model_name

        self.chat_template_name = chat_template_name
        self.chat_template = self.load_template(chat_template_name)

        self.hf_base_model_repo: str = hf_base_model_repo
        self.quantized_model_repos: Dict[QuantizationMethod, Tuple[str, str]] = quantized_model_repos

        self.hf_tokenizer: str = hf_base_model_repo
        self.use_hf_tokenizer: bool = use_hf_tokenizer

        self.optional: bool = optional

    def get_supported_quantization_methods(self):
        return self.quantized_model_repos.keys()


gguf_file_extension: str = "Q4_K_M.gguf"
default_filename: str = "model.safetensors"

model_mapping: Dict[str, ModelConfig] = {
    "llama-2-7b-chat": HFModelConfig(
        model_name="llama-2-7b-chat",
        chat_template_name="llama-2-chat",
        hf_base_model_repo="meta-llama/Llama-2-7b-chat-hf",
        quantized_model_repos={
            QuantizationMethod.GGUF: ("TheBloke/Llama-2-7B-Chat-GGUF", f"llama-2-7b-chat.{gguf_file_extension}"),
            QuantizationMethod.GPTQ: ("TheBloke/Llama-2-7B-Chat-GPTQ", default_filename),
            QuantizationMethod.AWQ: ("TheBloke/Llama-2-7B-Chat-AWQ", default_filename),
        },
        use_hf_tokenizer=True
    ),
    "llama-2-13b-chat": HFModelConfig(
        model_name="llama-2-13b-chat",
        chat_template_name="llama-2-chat",
        hf_base_model_repo="meta-llama/Llama-2-13b-chat-hf",
        quantized_model_repos={
            QuantizationMethod.GGUF: ("TheBloke/Llama-2-13B-Chat-GGUF", f"llama-2-13b-chat.{gguf_file_extension}"),
            QuantizationMethod.GPTQ: ("TheBloke/Llama-2-13B-Chat-GPTQ", default_filename),
            QuantizationMethod.AWQ: ("TheBloke/Llama-2-13B-Chat-AWQ", default_filename),
        },
        use_hf_tokenizer=True
    ),
    "llama-2-70b-chat": HFModelConfig(
        model_name="llama-2-70b-chat",
        chat_template_name="llama-2-chat",
        hf_base_model_repo="meta-llama/Llama-2-70b-chat-hf",
        quantized_model_repos={
            QuantizationMethod.GGUF: ("TheBloke/Llama-2-70B-Chat-GGUF", f"llama-2-70b-chat.{gguf_file_extension}"),
            QuantizationMethod.GPTQ: ("TheBloke/Llama-2-70B-Chat-GPTQ", default_filename),
            QuantizationMethod.AWQ: ("TheBloke/Llama-2-70B-Chat-AWQ", default_filename),
        },
        use_hf_tokenizer=True
    ),
    "orca-2-7b": HFModelConfig(
        model_name="orca-2-7b",
        chat_template_name="orca-2",
        hf_base_model_repo="microsoft/Orca-2-7b",
        quantized_model_repos={
            QuantizationMethod.GGUF: ("TheBloke/Orca-2-7B-GGUF", f"orca-2-7b.{gguf_file_extension}"),
            QuantizationMethod.GPTQ: ("TheBloke/Orca-2-7B-GPTQ", default_filename),
            QuantizationMethod.AWQ: ("TheBloke/Orca-2-7B-AWQ", default_filename),
        },
        use_hf_tokenizer=True
    ),
    "orca-2-13b": HFModelConfig(
        model_name="orca-2-13b",
        chat_template_name="orca-2",
        hf_base_model_repo="microsoft/Orca-2-7b",
        quantized_model_repos={
            QuantizationMethod.GGUF: ("TheBloke/Orca-2-13B-GGUF", f"orca-2-13b.{gguf_file_extension}"),
            QuantizationMethod.GPTQ: ("TheBloke/Orca-2-13B-GPTQ", default_filename),
            QuantizationMethod.AWQ: ("TheBloke/Orca-2-13B-AWQ", default_filename),
        },
        use_hf_tokenizer=True
    ),
    "phi-2": HFModelConfig(
        model_name="phi-2",
        chat_template_name="phi-2",
        hf_base_model_repo="microsoft/phi-2",
        quantized_model_repos={
            QuantizationMethod.GGUF: ("TheBloke/Phi-2-GGUF", f"phi-2.{gguf_file_extension}"),
            QuantizationMethod.GPTQ: ("TheBloke/Phi-2-GPTQ", default_filename),
            QuantizationMethod.AWQ: ("TheBloke/Phi-2-AWQ", default_filename),
        },
        use_hf_tokenizer=True
    ),
    "mistral-7b-instruct-v0.2": HFModelConfig(
        model_name="mistral-7b-instruct-v0.2",
        chat_template_name="mistral-7b-instruct",
        hf_base_model_repo="mistralai/Mistral-7B-Instruct-v0.2",
        quantized_model_repos={
            QuantizationMethod.GGUF: ("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", f"mistral-7b-instruct-v0.2.{gguf_file_extension}"),
            QuantizationMethod.GPTQ: ("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", default_filename),
            QuantizationMethod.AWQ: ("TheBloke/Mistral-7B-Instruct-v0.2-AWQ", default_filename),
        },
        use_hf_tokenizer=True
    ),


    "llama-3-8b": HFModelConfig(
        model_name="llama-3-8b",
        chat_template_name="fallback",
        hf_base_model_repo="meta-llama/Meta-Llama-3-8B",
        quantized_model_repos={
            QuantizationMethod.GGUF: ("QuantFactory/Meta-Llama-3-8B-GGUF", f"Meta-Llama-3-8B.{gguf_file_extension}"),
        },
        use_hf_tokenizer=True,
        optional=True
    ),
    "llama-3-8b-instruct": HFModelConfig(
        model_name="llama-3-8b-instruct",
        chat_template_name="llama-3-instruct",
        hf_base_model_repo="meta-llama/Meta-Llama-3-8B-Instruct",
        quantized_model_repos={
            QuantizationMethod.GGUF: ("QuantFactory/Meta-Llama-3-8B-Instruct-GGUF", f"Meta-Llama-3-8B-Instruct.{gguf_file_extension}"),
        },
        use_hf_tokenizer=True,
        optional=True
    ),
    "gemma-7b-it": HFModelConfig(
        model_name="gemma-7b-it",
        chat_template_name="gemma",
        hf_base_model_repo="google/gemma-7b-it",
        quantized_model_repos={
            QuantizationMethod.GGUF: ("mlabonne/gemma-7b-it-GGUF", f"gemma-7b-it.{gguf_file_extension}"),
        },
        use_hf_tokenizer=True,
        optional=True
    )
}