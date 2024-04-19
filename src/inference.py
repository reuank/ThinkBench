import os
from abc import abstractmethod, ABC
import time
from typing import List, Dict
from tqdm import tqdm

from llama_cpp import Llama, CreateCompletionResponse, LlamaGrammar
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

from benchmark import SingleBenchmarkResult
from completion_result import CompletionResult, Choice, Usage, Logprobs
from decoder import CompletionConfig, GreedyConstrainedDecoder, Decoder
from dataset import SingleDataInstance
from model import ModelConfig, HFModelConfig
from prompt import PromptChain, PromptCompletionStep, PromptTemplateStep
from testcase import TestCase, TestCaseResult


class MessageHistory:
    messages: [Dict[str, str]]
    user_role = "user"
    assistant_role = "assistant"

    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str):
        self.messages.append({"role": self.user_role, "content": message})

    def append_to_last_user_message(self, message: str):
        if len(self.messages) == 0 or self.messages[-1]["role"] != self.user_role:
            # there is no history so far, or last message was not a user message
            self.add_user_message(message)
        else:
            self.messages[-1]["content"] += "\n" + message

    def add_assistant_message(self, message: str):
        self.messages.append({"role": self.assistant_role, "content": message})

    def get_concatenated_messages(self):
        contents = [message["content"] for message in self.messages]

        return "\n".join(contents)

    def __repr__(self):
        return str(self.messages)


class InferenceBackend(ABC):
    current_model_config: ModelConfig
    loaded_model: None

    @property
    def name(self):
        return self.__class__.__name__

    @staticmethod
    def get_by_name(backend_name):
        inference_backend: InferenceBackend
        if backend_name == "llama-cpp-python" or backend_name == "default":
            inference_backend = LlamaCppPythonInferenceBackend()
        else:
            raise ValueError(f"Backend name '{backend_name}' not found.")

        return inference_backend

    @abstractmethod
    def load_model_from_config(self, model_config: ModelConfig):
        raise NotImplementedError

    @abstractmethod
    def create_completion(self, prompt: str, completion_config: CompletionConfig, decoder: Decoder) -> CompletionResult:
        raise NotImplementedError

    def run_test_case(self, test_case: TestCase, comment: str) -> TestCaseResult:
        start_time = time.time()

        print(test_case.get_info())

        test_data_instances = test_case.prepare_test_dataset()
        single_results: List[SingleBenchmarkResult] = []

        # run tests sequentially
        progressbar = tqdm(test_data_instances)
        progressbar.set_description("Benchmarking model")

        for single_test_data_instance in progressbar:
            prompt_chain_results = []

            for prompt_chain in test_case.benchmark.prompt_chains(single_test_data_instance):
                prompt_chain_results.append(self.execute_prompt_chain(prompt_chain, single_test_data_instance, test_case.use_chat_template))

                single_result = test_case.benchmark.compute_single_result(single_test_data_instance, prompt_chain_results)
                single_results.append(single_result)

        end_time = time.time()

        metrics = test_case.benchmark.compute_metrics(single_results)

        try:
            hostname = os.environ.get("TB_HOSTNAME")
            if not hostname:
                raise KeyError
        except KeyError:
            print("Please specify the necessary environment variables.")
            hostname = ""

        return TestCaseResult(
            model=self.current_model_config.model_name,
            dataset_name=test_case.dataset.name,
            benchmark_name=test_case.benchmark.name,
            label_numbering=test_case.label_numbering.value,
            hostname=hostname,
            inference_backend=self.name,
            inference_backend_properties=self.get_backend_properties(),
            metrics=metrics,
            start_time=start_time,
            end_time=end_time,
            execution_seconds=round(time.time() - start_time, 2),
            comment=comment,
            use_chat_template=test_case.use_chat_template,
            results=single_results
        )

    def execute_prompt_chain(self, prompt_chain: PromptChain, single_data_instance: SingleDataInstance, use_chat_template: bool) -> List[CompletionResult]:
        message_history = MessageHistory()
        completion_history: List[CompletionResult] = []

        # print(prompt_chain)

        for prompt_step in prompt_chain.steps:
            prompt_step_type = type(prompt_step)

            if prompt_step_type == PromptTemplateStep:
                prompt_step: PromptTemplateStep
                message = prompt_step.template.render(single_data_instance=single_data_instance)
                message_history.append_to_last_user_message(message)

            elif prompt_step_type == PromptCompletionStep:
                prompt_step: PromptCompletionStep

                if use_chat_template:
                    prompt = self.current_model_config.chat_template.render(
                        messages=message_history.messages,
                        bos_token=self.current_model_config.bos_token,
                        eos_token=self.current_model_config.eos_token,
                        add_generation_prompt=True
                    )
                else:
                    prompt = message_history.get_concatenated_messages()

                completion = self.create_completion(prompt, prompt_step.completion_config, prompt_step.decoder)
                completion_history.append(completion)

                message = str(completion.choices[0].text)
                message_history.add_assistant_message(message)
            else:
                raise ValueError(f"Prompt step type {prompt_step_type} not implemented.")

        return completion_history

    @abstractmethod
    def get_backend_properties(self):
        raise NotImplementedError


class LlamaCppPythonInferenceBackend(InferenceBackend):
    loaded_model: Llama
    n_gpu_layers: int = 1000
    n_ctx: int = 8192
    n_batch: int = 1024
    logits_all: bool = True
    verbose: bool = False

    def __init__(self):
        try:
            self.model_folder_path = os.environ.get("TB_MODEL_PATH")
            if not self.model_folder_path:
                raise KeyError
        except KeyError:
            print("Please specify a model path.")
            exit()

    def load_model_from_config(self, model_config: ModelConfig):
        if not isinstance(model_config, HFModelConfig):
            raise ValueError("Only HF Models are supported by this inference backend")

        model_config: HFModelConfig

        model_filename = f"{model_config.model_name}.Q4_K_M.gguf"
        model_path = f"{self.model_folder_path}/{model_filename}"

        if not os.path.isdir(self.model_folder_path):
            os.mkdir(self.model_folder_path)
            print("Model folder created, as it does not yet exist.")

        if not os.path.isfile(model_path):
            from huggingface_hub import hf_hub_download

            print(f"Download model {model_filename}, as it does not yet exist in the model folder.")
            hf_hub_download(
                repo_id=model_config.hf_repo,
                filename=model_filename,
                local_dir=self.model_folder_path,
                local_dir_use_symlinks=False
            )

        if type(model_config) != HFModelConfig:
            raise ValueError("Only HF models are supported by this inference backend so far.")

        model_config: HFModelConfig
        # Load correct Tokenizer
        if model_config.use_hf_tokenizer:
            tokenizer = LlamaHFTokenizer.from_pretrained(model_config.hf_tokenizer)
            print(f"External Tokenizer {model_config.hf_tokenizer} loaded.")
        else:
            tokenizer = None  # Defaults to LlamaTokenizer

        start_model_load = time.time()
        self.loaded_model: Llama = Llama(
            model_path=model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            logits_all=self.logits_all,
            tokenizer=tokenizer,
            verbose=self.verbose
        )
        end_model_load = time.time()

        print(f"Model {model_filename} loaded in {round(end_model_load - start_model_load, 2)} seconds.")

        self.current_model_config = model_config

        # llama_cpp_python seems to ignore special vocab when decoding:
        # tokenizer.decode([1, 8142, 417]) results in '<s> Hallo' with HF AutoTokenizer
        # However, self.loaded_model.tokenizer().decode([1, 15043, 2787]) just results in 'Hello World'
        # -> I need to explicitly add the bos and eos token to the model config
        # special_tokens = self.loaded_model.tokenizer().detokenize([self.loaded_model.token_bos(), self.loaded_model.token_eos()])
        # self.current_model_config.bos_token = special_tokens[0]
        # self.current_model_config.eos_token = special_tokens[1]

    def create_completion(self, prompt: str, completion_config: CompletionConfig, decoder: Decoder) -> CompletionResult:
        if type(decoder) == GreedyConstrainedDecoder:
            decoder: GreedyConstrainedDecoder

            grammar = LlamaGrammar.from_string(
                grammar=self.get_grammar_string_from_labels(decoder.allowed_strings),
                verbose=False
            )

            completion_response: CreateCompletionResponse = self.loaded_model.create_completion(
                prompt=prompt,
                max_tokens=completion_config.max_tokens,
                temperature=completion_config.temperature,
                logprobs=completion_config.max_logprobs,
                grammar=grammar
            )

            return self.__convert_completion_response(prompt, completion_response)

    @staticmethod
    def __convert_completion_response(prompt: str, completion_response: CreateCompletionResponse) -> CompletionResult:
        choices = [Choice(
            choice["text"],
            choice["index"],
            Logprobs(
                completion_response['choices'][choice_id]["logprobs"]["tokens"],
                completion_response['choices'][choice_id]["logprobs"]["text_offset"],
                completion_response['choices'][choice_id]["logprobs"]["token_logprobs"],
                completion_response['choices'][choice_id]["logprobs"]["top_logprobs"]
            ),
            choice["finish_reason"]
        ) for choice_id, choice in enumerate(completion_response['choices'])]

        usage = Usage(
            completion_response["usage"]["prompt_tokens"],
            completion_response["usage"]["completion_tokens"],
            completion_response["usage"]["total_tokens"]
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
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "logits_all": self.logits_all,
            "verbose": self.verbose
        }

