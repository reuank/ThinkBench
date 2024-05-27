from typing import Tuple, Dict

from model_config.model_config import ModelConfig, QuantizationMethod


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
