import pathlib
import uuid

import git
import math
import os
import sys
import signal
import threading
import time
from string import Template
from abc import abstractmethod, ABC
from pathlib import Path
from queue import Queue
from subprocess import Popen
from typing import List, Dict, Any

import psutil
from requests import Session
from tqdm import tqdm

from llama_cpp import Llama, CreateCompletionResponse, LlamaGrammar
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

from benchmark import SingleBenchmarkResult
from completion import CompletionResult, Choice, Usage, Logprobs, CompletionHistory, CompletionConfig
from decoder import GreedyConstrainedDecoder, Decoder, GreedyDecoder, BeamSearch, TemperatureDecoder
from dataset import SingleDataInstance
from model import ModelConfig, HFModelConfig, QuantizationMethod
from prompt import PromptChain, PromptCompletionStep, PromptTemplateStep, PromptTextStep
from testcase import TestCase, TestCaseResult
from utils.timer import Timer


server_debug = False


class MessageHistory:
    messages: [Dict[str, str]]
    user_role = "user"
    assistant_role = "assistant"

    def __init__(self):
        self.messages = []

    def __repr__(self):
        return str(self.messages)

    def add_user_message(self, message: str):
        self.messages.append({"role": self.user_role, "content": message})

    def append_to_last_user_message(self, message: str):
        if len(self.messages) == 0 or self.messages[-1]["role"] != self.user_role:
            # there is no history so far, or last message was not a user message
            self.add_user_message(message)
        else:
            self.messages[-1]["content"] += message

    def add_assistant_message(self, message: str):
        self.messages.append({"role": self.assistant_role, "content": message})

    def get_concatenated_messages(self):
        contents = [message["content"] for message in self.messages]

        return "\n".join(contents)


class InferenceBackend(ABC):
    current_model_config: ModelConfig
    loaded_model: None
    verbose: bool = False

    @property
    def name(self):
        return self.__class__.__name__

    @property
    @abstractmethod
    def supported_quantization_methods(self) -> List[QuantizationMethod]:
        raise NotImplementedError

    @staticmethod
    def get_by_name(backend_name):
        inference_backend: InferenceBackend
        if backend_name == "llama-cpp-python" or backend_name == "default":
            inference_backend = LlamaCppPythonInferenceBackend()
        elif backend_name == "llama.cpp":
            inference_backend = LlamaCppServerInferenceBackend()
        elif backend_name == "llama.cpp-multi-gpu":
            inference_backend = LlamaCppMultiGPUServerInferenceBackend()
        elif backend_name == "transformers":
            inference_backend = TransformersInferenceBackend()
        else:
            raise ValueError(f"Backend name '{backend_name}' not found.")

        return inference_backend

    @staticmethod
    def ensure_hf_model_is_downloaded(local_path: Path, hf_repo: str, model_filename: str):
        if not os.path.isfile(local_path/model_filename):
            from huggingface_hub import hf_hub_download

            print(f"Download model {model_filename}, as it does not yet exist in the model folder.")
            hf_hub_download(
                repo_id=hf_repo,
                filename=model_filename,
                local_dir=local_path,
                local_dir_use_symlinks=False
            )

    @abstractmethod
    def load_model_from_config(self, model_config: ModelConfig):
        raise NotImplementedError

    @abstractmethod
    def create_completion(self, prompt: str, completion_config: CompletionConfig, decoder: Decoder, additional_params: Dict[str, Any]) -> CompletionResult:
        raise NotImplementedError

    @abstractmethod
    def _run_test_case(self, test_case: TestCase, test_data_instances: List[SingleDataInstance]) -> List[SingleBenchmarkResult]:
        raise NotImplementedError

    def run_test_case(self, test_case: TestCase, comment: str) -> TestCaseResult:
        Timer.get_instance("test_case").start_over()

        print(test_case.get_info())

        single_results: List[SingleBenchmarkResult] = self._run_test_case(
            test_case=test_case,
            test_data_instances=test_case.prepare_test_dataset()
        )

        metrics = test_case.benchmark.compute_metrics(single_results)

        try:
            hostname = os.environ.get("TB_HOSTNAME")
            if not hostname:
                raise KeyError
        except KeyError:
            print("Please specify the necessary environment variables.")
            hostname = ""

        print("="*45)
        Timer.get_instance("test_case").end()
        print(f"Metrics: {metrics}.")

        repo = git.Repo(path=pathlib.Path(__file__).parent.resolve(), search_parent_directories=True)
        current_commit_hash = repo.head.object.hexsha[:8]

        return TestCaseResult(
            uuid=str(uuid.uuid4()),
            model=self.current_model_config.model_name,
            dataset_name=test_case.dataset.name,
            benchmark_name=test_case.benchmark.name,
            label_numbering=test_case.label_numbering.value,
            label_permutation=test_case.label_permutation.value,
            n_random=test_case.n_random,
            random_seed=test_case.random_seed,
            hostname=hostname,
            inference_backend=self.name,
            inference_backend_properties=self.get_backend_properties(),
            metrics=metrics,
            start_time=Timer.get_instance("test_case").start_time,
            end_time=Timer.get_instance("test_case").end_time,
            execution_seconds=Timer.get_instance("test_case").elapsed_time,
            current_commit_hash=current_commit_hash,
            comment=comment,
            use_chat_template=test_case.use_chat_template,
            results=single_results
        )

    def execute_prompt_chain(self, prompt_chain: PromptChain, single_data_instance: SingleDataInstance, use_chat_template: bool, additional_params: Dict[str, Any]) -> CompletionHistory:
        message_history = MessageHistory()
        completion_history = CompletionHistory()

        # print(prompt_chain)

        for prompt_step in prompt_chain.steps:
            prompt_step_type = type(prompt_step)

            if prompt_step_type == PromptTextStep:
                prompt_step: PromptTextStep
                message = prompt_step.text
                message_history.append_to_last_user_message(message)

            elif prompt_step_type == PromptTemplateStep:
                prompt_step: PromptTemplateStep
                message = prompt_step.template.render(single_data_instance=single_data_instance)
                message_history.append_to_last_user_message(message)

            elif prompt_step_type == PromptCompletionStep:
                prompt_step: PromptCompletionStep

                if use_chat_template:
                    unfilled_prompt = self.current_model_config.chat_template.render(
                        messages=message_history.messages,
                        add_generation_prompt=True
                    )
                else:
                    unfilled_prompt = message_history.get_concatenated_messages()

                previous_completion_texts = completion_history.get_texts()

                try:
                    filled_prompt = Template(unfilled_prompt).safe_substitute(previous_completion_texts)
                except ValueError:
                    raise ValueError("Provided prompt variables not correct.")

                completion_timer_name = f"Completion Q={single_data_instance.id} Step={prompt_step.name}"
                Timer.get_instance(completion_timer_name).start_over()
                completion_result = self.create_completion(
                    prompt=filled_prompt,
                    completion_config=prompt_step.completion_config,
                    decoder=prompt_step.decoder,
                    additional_params=additional_params
                )
                completion_result.prompt = unfilled_prompt  # remove model generated content from prompt
                Timer.get_instance(completion_timer_name).end()

                completion_history.add_completion(
                    name=prompt_step.name,
                    completion_result=completion_result,
                    completion_config=prompt_step.completion_config,
                    decoder=prompt_step.decoder
                )

                message = "${" + prompt_step.name + "}"
                message_history.add_assistant_message(prompt_step.prefix + message + prompt_step.suffix)

            else:
                raise ValueError(f"Prompt step type {prompt_step_type} not implemented.")

        return completion_history

    def set_verbosity(self, verbose: bool):
        self.verbose = verbose

    @abstractmethod
    def get_backend_properties(self):
        raise NotImplementedError


class LlamaCppPythonInferenceBackend(InferenceBackend):
    loaded_model: Llama = None
    n_gpu_layers: int = 1000
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
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            logits_all=self.logits_all,
            tokenizer=tokenizer,
            verbose=self.verbose
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
            temperature=completion_config.temperature,
            logprobs=completion_config.max_logprobs,
            grammar=grammar,
            repeat_penalty=completion_config.repeat_penalty
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
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "logits_all": self.logits_all,
            "verbose": self.verbose
        }


class LlamaCppServerInferenceBackend(InferenceBackend):
    n_gpu_layers: int = 1000
    n_ctx: int = 8192
    n_batch: int = 4096
    n_parallel: int = 1
    continuous_batching: bool = False
    tokenize_before: bool = True

    process: Popen = None
    completion_url_template: Template = Template("http://localhost:${port}/completion")
    tokenization_url_template: Template = Template("http://localhost:${port}/tokenize")
    properties_url_template: Template = Template("http://localhost:${port}/props")
    headers = {'content-type': 'application/json'}

    @property
    def current_commit_hash(self):
        repo = git.Repo(path=self.server_binary_path, search_parent_directories=True)
        current_commit_hash = repo.head.object.hexsha[:8]

        return current_commit_hash

    @property
    def name(self):
        return f"{self.__class__.__name__}-{self.current_commit_hash}"

    @property
    def supported_quantization_methods(self) -> List[QuantizationMethod]:
        return [QuantizationMethod.GGUF]

    def __init__(self):
        # TODO: implement ensure_exists() function or python config file
        try:
            model_folder_path_str = os.environ.get("TB_MODEL_PATH")
            n_parallel = os.environ.get("TB_LLAMA_CPP_SERVER_SLOTS")
            server_binary_path = os.environ.get("LLAMA_CPP_SERVER_BINARY")
            port = os.environ.get("LLAMA_CPP_SERVER_PORT")
            if not model_folder_path_str or not n_parallel or not server_binary_path:
                raise KeyError
            else:
                self.model_folder_path: Path = Path(model_folder_path_str)
                self.model_folder_path.mkdir(parents=True, exist_ok=True)
                self.n_parallel = int(n_parallel)
                self.n_ctx = self.n_parallel * 4096
                self.server_binary_path: Path = Path(server_binary_path)

                if not port:
                    port = 8080
                self.port: int = int(port)
        except KeyError:
            print("Please specify a model path, the number of server slots and the server binary path. Did you forget to source .env?")
            exit()

        self.session: Session = Session()

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

        # spawn a new server
        Timer.get_instance(f"Load {hf_filename}").start_over(print_out=True)
        server_process_arguments = [
            str(self.server_binary_path),
            "--port", str(self.port),
            "-m", str(self.model_folder_path/hf_filename),
            "-b", str(self.n_batch),
            "-c", str(self.n_ctx),
            "-ngl", str(self.n_gpu_layers),
            "-np", str(self.n_parallel),
            "--log-disable"
        ]
        if self.continuous_batching:
            server_process_arguments.append("-cb")

        if server_debug:
            stdout = sys.stdout
        else:
            stdout = subprocess.DEVNULL

        self.process = subprocess.Popen(server_process_arguments, stdout=stdout, stderr=subprocess.STDOUT)
        time.sleep(2)
        print(f"Currently loaded model on the available server: {self.get_backend_properties()['loaded_model']}")
        Timer.get_instance(f"Load {hf_filename}").end(print_out=True)

        self.current_model_config = model_config

    def _run_test_case_subset(self, test_case: TestCase, thread_id: int, test_data_subset: List[SingleDataInstance], output_queue: Queue, shared_progressbar: tqdm):
        for single_test_data_instance in test_data_subset:
            prompt_chain_results = []

            for prompt_chain in test_case.benchmark.prompt_chains(single_test_data_instance):
                prompt_chain_completion_history = self.execute_prompt_chain(
                    prompt_chain=prompt_chain,
                    single_data_instance=single_test_data_instance,
                    use_chat_template=test_case.use_chat_template,
                    additional_params={
                        "id_slot": thread_id  # ensure that a thread only uses its own server slot
                    }
                )
                prompt_chain_results.append(prompt_chain_completion_history)

            single_result = test_case.benchmark.compute_single_result(single_test_data_instance, prompt_chain_results)
            output_queue.put(single_result)
            shared_progressbar.update(1)

    def _run_test_case(self, test_case: TestCase, test_data_instances: List[SingleDataInstance]) -> List[SingleBenchmarkResult]:
        threads = []
        output_queue = Queue()

        def distribute_chunks(data, num_threads):
            n = len(data)
            chunk_size = n // num_threads
            remainder = n % num_threads

            chunks = []
            start = 0

            for thread_id in range(num_threads):
                end = start + chunk_size + (1 if thread_id < remainder else 0)
                chunks.append(data[start:end])
                start = end

            # print(f"{chunk_size=} {[len(chunk) for chunk in chunks]=}")

            return chunks

        chunks = distribute_chunks(data=test_data_instances, num_threads=self.n_parallel)
        shared_progressbar = tqdm(total=len(test_data_instances), desc=f"Benchmarking model on {self.n_parallel} server slots.")

        for i in range(self.n_parallel):
            thread = threading.Thread(target=self._run_test_case_subset, args=(test_case, i, chunks[i], output_queue, shared_progressbar))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        shared_progressbar.close()

        all_results: List[SingleBenchmarkResult] = []
        while not output_queue.empty():
            all_results.append(output_queue.get())

        all_results.sort(key=lambda single_result: single_result["question_id"])

        return all_results

    def create_completion(self, prompt: str, completion_config: CompletionConfig, decoder: Decoder, additional_params: Dict[str, Any]) -> CompletionResult:
        samplers = []
        if type(decoder) == GreedyConstrainedDecoder:
            completion_config.temperature = -1.0  # Return probs even when using greedy decoding
            samplers = ["temperature"]
        elif type(decoder) == TemperatureDecoder:
            decoder: TemperatureDecoder
            completion_config.temperature = decoder.temperature
            samplers = ["temperature"]

        if self.tokenize_before:
            prompt = self.session.post(
                url=self.tokenization_url_template.substitute(port=self.port),
                headers=self.headers,
                json={
                    "content": prompt,
                    "add_special": False
                }
            ).json()["tokens"]

        request = {
            "prompt": prompt,
            "id_slot": additional_params["id_slot"],  # ensure that a thread only uses its own server slot
            "n_predict": completion_config.max_tokens,
            "n_probs": completion_config.max_logprobs,
            "min_keep": completion_config.max_logprobs,
            "temperature": completion_config.temperature,
            "samplers": samplers,
            "seed": 1234,
            "repeat_last_n": 0,
            "min_p": 0.0,
            "top_p": 1.0,
            "top_k": 100,
            "repeat_penalty": completion_config.repeat_penalty,
            "mirostat_eta": 0.0,
            "mirostat_tau": 0.0,
            "cache_prompt": True
            # "grammar": grammar_string
        }

        raw_completion_response = self.session.post(
            url=self.completion_url_template.substitute(port=self.port),
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

    @staticmethod
    def __convert_completion_response(prompt: str, completion_response: Dict) -> CompletionResult:
        def get_logprob(prob: float) -> float:
            return prob
            return math.log(prob) if prob != 0 else -100.0

        finish_reason = ""
        if completion_response["stopped_eos"]:
            finish_reason = "eos"
        elif completion_response["stopped_word"]:
            finish_reason = "stop"
        elif completion_response["stopped_limit"]:
            finish_reason = "length"

        tokens: List[str] = []
        token_logprobs: List[float] = []
        top_logprobs: List[Dict[str, float]] = []

        # Example for completion_probabilities:
        # "completion_probabilities": [
        #     {
        #       "content": "B",
        #       "probs": [
        #         {
        #           "tok_str": "B",
        #           "prob": 0.44181621074676514
        #         },
        #         {
        #           "tok_str": "C",
        #           "prob": 0.37636300921440125
        #         }
        #       ]
        #     },
        #     {
        #       "content": "",
        #       "probs": [
        #         {
        #           "tok_str": "",
        #           "prob": 1.0
        #         },
        #         {
        #           "tok_str": " deleg",
        #           "prob": 0.0
        #         }
        #       ]
        for completion_probability in completion_response["completion_probabilities"]:
            top_logprobs.append(
                {item["tok_str"]: get_logprob(item["prob"]) for item in completion_probability["probs"]}
            )
            tokens.append(completion_probability["content"])
            token_logprobs.append(get_logprob(completion_probability["probs"][0]["prob"]))

        choices = [Choice(
            text=completion_response["content"],
            index=0,
            logprobs=Logprobs(
                tokens=tokens,
                text_offset=[],
                token_logprobs=[],
                top_logprobs=top_logprobs
            ),
            finish_reason=finish_reason
        )]

        timings = completion_response["timings"]
        usage = Usage(
            prompt_tokens=timings["prompt_n"],
            prompt_tokens_per_second=timings["prompt_per_second"],
            prompt_ms=timings["prompt_ms"],
            completion_tokens=timings["predicted_n"],
            completion_tokens_per_second=timings["predicted_per_second"],
            completion_ms=timings["predicted_ms"],
            total_tokens=timings["prompt_n"]+timings["predicted_n"]
        )

        return CompletionResult(
            prompt=prompt,
            id=completion_response['id_slot'],
            object="text_completion",
            created=int(time.time()),
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

    def get_backend_properties(self) -> Dict[str, str]:
        server_settings = self.session.get(url=self.properties_url_template.substitute(port=self.port), headers=self.headers).json()

        return {
            "loaded_model": server_settings["default_generation_settings"]["model"],
            "n_slots": server_settings["total_slots"],
            "n_ctx_per_slot": server_settings["default_generation_settings"]["n_ctx"],
            "batch_size": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "continuous_batching": self.continuous_batching,
            "current_commit_hash": self.current_commit_hash
        }

    def terminate_all_running_servers(self):
        program_name = str(self.server_binary_path)

        """ Kills all processes that contain the program_name in their executable path. """
        for proc in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                # Check if process name or the executable matches the program name
                if program_name in proc.info['name'] or (proc.info['exe'] and program_name in proc.info['exe']):
                    print(f"Killing process '{proc.info['name']}' with PID {proc.info['pid']}")
                    proc.send_signal(signal.SIGTERM)  # or proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass  # Process has been killed or can't be accessed


class LlamaCppMultiGPUServerInferenceBackend(LlamaCppServerInferenceBackend):
    def __init__(self):
        # TODO: implement ensure_exists() function or python config file
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
                "-ngl", str(self.n_gpu_layers),
                "-np", str(self.n_parallel),
                "--log-disable"
            ]
            if self.continuous_batching:
                server_process_arguments.append("-cb")

            if server_debug:
                stdout = sys.stdout
            else:
                stdout = subprocess.DEVNULL

            self.process = subprocess.Popen(server_process_arguments, shell=True, stdout=stdout, stderr=subprocess.STDOUT)
            time.sleep(2)
            print(f"Currently loaded model on the available server: {self.get_backend_properties()['loaded_model']}")

        Timer.get_instance(f"Load {hf_filename}").end(print_out=True)

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
            "seed": 1234,
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
            url=self.completion_url_template.substitute(port=self.port + additional_params["id_slot"]),
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


class TransformersInferenceBackend(InferenceBackend):
    loaded_model: None

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
        return [QuantizationMethod.AWQ, QuantizationMethod.GPTQ]

    def load_model_from_config(self, model_config: ModelConfig):
        raise NotImplementedError

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