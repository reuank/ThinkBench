import os
from pathlib import Path
from typing import List, Dict, Any

from llama_cpp.llama_tokenizer import LlamaHFTokenizer

from benchmark.benchmark import SingleBenchmarkResult
from dataset.single_data_instance import SingleDataInstance
from inference.completion import CompletionConfig, CompletionResult
from inference.decoder import Decoder
from inference.inference_backend import InferenceBackend, INFERENCE_BACKEND_REGISTRY
from benchmark.testcase import TestCase
from model_config.hf_model_config import HFModelConfig
from model_config.model_config import QuantizationMethod, ModelConfig
from utils.timer import Timer


@INFERENCE_BACKEND_REGISTRY.register("transformers")
class TransformersInferenceBackend(InferenceBackend):
    loaded_model: None

    def __init__(self):
        raise NotImplementedError

        try:
            model_folder_path_str = os.environ.get("TB_MODEL_PATH")
            if not model_folder_path_str:
                raise KeyError
            else:
                self.model_folder_path: Path = Path(model_folder_path_str)
                self.model_folder_path.mkdir(parents=True, exist_ok=True)
        except KeyError:
            print("Please specify a model path. Did you forget to source .env?")
            exit()

    @property
    def supported_quantization_methods(self) -> List[QuantizationMethod]:
        return [QuantizationMethod.AWQ, QuantizationMethod.GPTQ]

    def load_model_from_config(self, model_config: ModelConfig):
        from transformers import AutoModelForCausalLM
        if self.loaded_model:
            del self.loaded_model

        if not isinstance(model_config, HFModelConfig):
            raise ValueError("Only HF Models are supported by this inference backend")

        model_config: HFModelConfig
        intersection = list(set(model_config.get_supported_quantization_methods()) & set(self.supported_quantization_methods))
        if not intersection:
            raise ValueError(f"This backend supports quantization methods {self.supported_quantization_methods},"
                             f", but model {model_config.model_name} does only specify repos for methods "
                             f"{model_config.get_supported_quantization_methods()}")

        # Use first intersecting quantization method by default
        hf_repo, hf_filename = model_config.quantized_model_repos[intersection[0]]

        InferenceBackend.ensure_hf_model_is_downloaded(local_path=self.model_folder_path, hf_repo=hf_repo, model_filename=hf_filename)

        # Load correct Tokenizer
        if model_config.use_hf_tokenizer:
            tokenizer = LlamaHFTokenizer.from_pretrained(model_config.hf_tokenizer)
            print(f"External Tokenizer {model_config.hf_tokenizer} loaded.")
        else:
            tokenizer = None  # Defaults to LlamaTokenizer

        Timer.get_instance(f"Load {hf_filename}").start_over()
        self.loaded_model = AutoModelForCausalLM.from_pretrained(hf_repo)
        Timer.get_instance(f"Load {hf_filename}").end()

        self.current_model_config = model_config

    def create_completion(self, prompt: str, completion_config: CompletionConfig, decoder: Decoder,
                          additional_params: Dict[str, Any]) -> CompletionResult:
        pass

    def _run_test_case(self, test_case: TestCase, test_data_instances: List[SingleDataInstance]) -> List[SingleBenchmarkResult]:
        pass

    def get_backend_properties(self):
        pass
