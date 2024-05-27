import os
from pathlib import Path
from typing import List, Dict, Any

from llama_cpp import Llama, LlamaGrammar, CreateCompletionResponse
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
from tqdm import tqdm

from benchmark.benchmark import SingleBenchmarkResult
from constants import N_GPU_LAYERS, INFERENCE_BACKEND_VERBOSE
from dataset.single_data_instance import SingleDataInstance
from inference.completion import CompletionHistory, CompletionConfig, CompletionResult, Choice, Logprobs, Usage
from inference.decoder import Decoder, GreedyConstrainedDecoder, GreedyDecoder, BeamSearch
from inference.inference_backend import InferenceBackend, INFERENCE_BACKEND_REGISTRY
from benchmark.testcase import TestCase
from model_config.hf_model_config import HFModelConfig
from model_config.model_config import ModelConfig, QuantizationMethod
from utils.timer import Timer


@INFERENCE_BACKEND_REGISTRY.register("llama-cpp-python", is_default=True)
class LlamaCppPythonInferenceBackend(InferenceBackend):
    loaded_model: Llama = None
    n_ctx: int = 8192
    n_batch: int = 1024
    logits_all: bool = True

    def __init__(self):
        # TODO: implement ensure_exists() function or python config file
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
        return [QuantizationMethod.GGUF]

    def load_model_from_config(self, model_config: ModelConfig):
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
        self.loaded_model: Llama = Llama(
            model_path=str(self.model_folder_path/hf_filename),
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            logits_all=self.logits_all,
            tokenizer=tokenizer,
            verbose=INFERENCE_BACKEND_VERBOSE
        )
        Timer.get_instance(f"Load {hf_filename}").end()

        self.current_model_config = model_config

        # llama_cpp_python seems to ignore special vocab when decoding:
        # tokenizer.decode([1, 8142, 417]) results in '<s> Hallo' with HF AutoTokenizer
        # However, self.loaded_model.tokenizer().decode([1, 15043, 2787]) just results in 'Hello World'
        # -> I need to explicitly add the bos and eos token to the model config
        # special_tokens = self.loaded_model.tokenizer().detokenize([self.loaded_model.token_bos(), self.loaded_model.token_eos()])
        # self.current_model_config.bos_token = special_tokens[0]
        # self.current_model_config.eos_token = special_tokens[1]

    def _run_test_case(self, test_data_instances: List[SingleDataInstance], test_case: TestCase) -> List[SingleBenchmarkResult]:
        single_results: List[SingleBenchmarkResult] = []
        # run tests sequentially
        progressbar = tqdm(test_data_instances)
        progressbar.set_description("Benchmarking model")

        for single_test_data_instance in progressbar:
            prompt_chain_results: List[CompletionHistory] = []

            for prompt_chain in test_case.benchmark.prompt_chains(single_test_data_instance):
                prompt_chain_completion_history = self.execute_prompt_chain(
                    prompt_chain=prompt_chain,
                    single_data_instance=single_test_data_instance,
                    use_chat_template=test_case.use_chat_template,
                    additional_params={}
                )
                prompt_chain_results.append(prompt_chain_completion_history)

            single_result = test_case.benchmark.compute_single_result(single_test_data_instance, prompt_chain_results)
            single_results.append(single_result)

        return single_results

    def create_completion(self, prompt: str, completion_config: CompletionConfig, decoder: Decoder, additional_params: Dict[str, Any]) -> CompletionResult:
        grammar = None

        if type(decoder) == GreedyConstrainedDecoder:
            decoder: GreedyConstrainedDecoder

            grammar_string = self.get_grammar_string_from_labels(decoder.allowed_strings)

            grammar = LlamaGrammar.from_string(
                grammar=grammar_string,
                verbose=False
            )

        if type(decoder) == GreedyDecoder:  # TODO
            pass

        if type(decoder) == BeamSearch:  # TODO
            pass

        completion_response: CreateCompletionResponse = self.loaded_model.create_completion(
            prompt=prompt,
            max_tokens=completion_config.max_tokens,
            temperature=decoder.temperature,
            logprobs=completion_config.max_logprobs,
            grammar=grammar,
            repeat_penalty=decoder.repeat_penalty
        )

        return self.__convert_completion_response(prompt, completion_response)

    @staticmethod
    def __convert_completion_response(prompt: str, completion_response: CreateCompletionResponse) -> CompletionResult:
        choices = [Choice(
            text=choice["text"],
            index=choice["index"],
            logprobs=Logprobs(
                tokens=completion_response['choices'][choice_id]["logprobs"]["tokens"],
                text_offset=completion_response['choices'][choice_id]["logprobs"]["text_offset"],
                token_logprobs=completion_response['choices'][choice_id]["logprobs"]["token_logprobs"],
                top_logprobs=completion_response['choices'][choice_id]["logprobs"]["top_logprobs"]
            ),
            finish_reason=choice["finish_reason"]
        ) for choice_id, choice in enumerate(completion_response['choices'])]

        usage = Usage(
            prompt_tokens=completion_response["usage"]["prompt_tokens"]-completion_response["usage"]["completion_tokens"],
            prompt_tokens_per_second=0,
            prompt_ms=0,
            completion_tokens=completion_response["usage"]["completion_tokens"],
            completion_tokens_per_second=0,
            completion_ms=0,
            total_tokens=completion_response["usage"]["total_tokens"]
        ) if 'usage' in completion_response else None

        return CompletionResult(
            prompt=prompt,
            id=completion_response['id'],
            object=completion_response['object'],
            created=completion_response['created'],
            model=completion_response['model'],
            choices=choices,
            usage=usage
        )

    @staticmethod
    def get_grammar_string_from_labels(labels: [str]) -> str:
        # labels_with_quotes = list(map(lambda label: f'"{label}"', labels))

        # Accept only label tokens, e.g. "A", " A", ...
        return f'root ::= " "? [{labels[0]}-{labels[-1]}]'
        # return f"root   ::= [ ]? option \noption ::= ({'|'.join(labels_with_quotes)})"

    def get_backend_properties(self):
        return {
            "n_gpu_layers": N_GPU_LAYERS,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "logits_all": self.logits_all,
            "verbose": self.verbose
        }