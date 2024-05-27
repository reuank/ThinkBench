import pathlib
from abc import ABC
from enum import Enum

from jinja2 import Template, FileSystemLoader, Environment

from constants import LIBRARY_ROOT
from utils.registry import Registry


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
        template_loader = FileSystemLoader(searchpath=LIBRARY_ROOT / "chat_templates")
        template_env = Environment(loader=template_loader, trim_blocks=True, lstrip_blocks=True)
        template_file = f"{template_name}.jinja"

        return template_env.get_template(template_file)


MODEL_CONFIG_REGISTRY = Registry(
    registry_name="model_configs",
    base_class=ModelConfig,
    lazy_load_dirs=["model_config/model_configs"]
)
