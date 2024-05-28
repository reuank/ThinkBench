import sys
import time
from typing import Any, Dict

from constants import N_GPU_LAYERS, COMPLETION_SEED, SERVER_HOST
from inference.backends.llama_cpp_server_backend import LlamaCppServerInferenceBackend
from inference.completion import CompletionConfig, CompletionResult
from inference.decoder import GreedyConstrainedDecoder, Decoder
from inference.inference_backend import INFERENCE_BACKEND_REGISTRY, InferenceBackend
from model_config.hf_model_config import HFModelConfig
from model_config.model_config import ModelConfig
from utils.timer import Timer


@INFERENCE_BACKEND_REGISTRY.register("llama.cpp-multi-gpu")
class LlamaCppMultiGPUServerInferenceBackend(LlamaCppServerInferenceBackend):
    def __init__(self):
        raise NotImplementedError
        super().__init__()
        self.n_ctx = 4096

    def load_model_from_config(self, model_config: ModelConfig):
        import subprocess

        print("Terminating any old server processes...")
        if self.process:
            self.process.terminate()
        self.terminate_all_running_servers()
        time.sleep(2)

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

        Timer.get_instance(f"Load {hf_filename}").start_over(print_out=True)
        # spawn n_parallel new servers
        for i in range(self.n_parallel):
            print(f"Starting server on port {self.port + i} for GPU id {i}")
            server_process_arguments = [
                f"export CUDA_VISIBLE_DEVICES={i};",
                str(self.server_binary_path),
                "--port", str(self.port + i),  # 8080, 8081, 8082...
                "-m", str(self.model_folder_path/hf_filename),
                "-b", str(self.n_batch),
                "-c", str(self.n_ctx),
                "-ngl", str(N_GPU_LAYERS),
                "-np", str(self.n_parallel),
                "--log-disable"
            ]
            if self.continuous_batching:
                server_process_arguments.append("-cb")

            if self.server_debug:
                stdout = sys.stdout
            else:
                stdout = subprocess.DEVNULL

            self.process = subprocess.Popen(server_process_arguments, shell=True, stdout=stdout, stderr=subprocess.STDOUT)
            time.sleep(2)
            print(f"Currently loaded model on the available server: {self.get_backend_properties()['loaded_model']}")

        Timer.get_instance(f"Load {hf_filename}").end(print_timer=True)

        self.current_model_config = model_config

    def create_completion(self, prompt: str, completion_config: CompletionConfig, decoder: Decoder, additional_params: Dict[str, Any]) -> CompletionResult:
        if type(decoder) == GreedyConstrainedDecoder:
            completion_config.temperature = -1.0  # Return probs even when using greedy decoding

        request = {
            "prompt": prompt,
            "n_predict": completion_config.max_tokens,
            "n_probs": completion_config.max_logprobs,
            "min_keep": completion_config.max_logprobs,
            "temperature": completion_config.temperature,
            "samplers": ["temperature"],
            "seed": COMPLETION_SEED,
            "repeat_last_n": 0,
            "min_p": 0.0,
            "top_p": 1.0,
            "top_k": 100,
            "repeat_penalty": 1.0,
            "mirostat_eta": 0.0,
            "mirostat_tau": 0.0,
            "cache_prompt": True
            # "grammar": grammar_string
        }

        raw_completion_response = self.session.post(
            url=self.completion_url_template.substitute(port=self.port + additional_params["id_slot"], host=SERVER_HOST),
            headers=self.headers,
            json=request).json()
        completion_response = self.__convert_completion_response(prompt, raw_completion_response)

        if type(decoder) == GreedyConstrainedDecoder:
            # Using a grammar directly in llama.cpp does seem to work differently as compared to llama-cpp-python.
            decoder: GreedyConstrainedDecoder

            allowed_tokens = []
            for allowed_token in decoder.allowed_strings:
                allowed_tokens.append(f"{allowed_token}")
                allowed_tokens.append(f" {allowed_token}")

            tokens: Dict[str, float] = completion_response.choices[0].logprobs.top_logprobs[0]
            sorted_tokens = {k: v for k, v in sorted(tokens.items(), key=lambda item: item[1], reverse=True)}
            filtered_tokens = {k: v for k, v in filter(lambda item: item[0] in allowed_tokens, tokens.items())}

            if filtered_tokens == {}:
                selection = "NONE"
            else:
                selection = next(iter(filtered_tokens.keys()))

            completion_response.choices[0].text = selection
            completion_response.choices[0].logprobs.top_logprobs[0] = sorted_tokens

        return completion_response