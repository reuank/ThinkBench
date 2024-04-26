import json
import math
import os
import string
import threading
import time
from abc import abstractmethod, ABC
from pathlib import Path
from queue import Queue
from subprocess import Popen
from typing import List, Dict, Any

from requests import Session
from tqdm import tqdm

from llama_cpp import Llama, CreateCompletionResponse, LlamaGrammar
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

from benchmark import SingleBenchmarkResult
from completion import CompletionResult, Choice, Usage, Logprobs, CompletionHistory, CompletionConfig
from decoder import GreedyConstrainedDecoder, Decoder, GreedyDecoder, BeamSearch
from dataset import SingleDataInstance
from model import ModelConfig, HFModelConfig
from prompt import PromptChain, PromptCompletionStep, PromptTemplateStep, PromptTextStep
from storage import TotalResultEncoder
from testcase import TestCase, TestCaseResult
from utils.timer import Timer


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
            self.messages[-1]["content"] += "\n" + message

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

    @staticmethod
    def get_by_name(backend_name):
        inference_backend: InferenceBackend
        if backend_name == "llama-cpp-python" or backend_name == "default":
            inference_backend = LlamaCppPythonInferenceBackend()
        elif backend_name == "llama.cpp":
            inference_backend = LlamaCppServerInferenceBackend()
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

        return TestCaseResult(
            model=self.current_model_config.model_name,
            dataset_name=test_case.dataset.name,
            benchmark_name=test_case.benchmark.name,
            label_numbering=test_case.label_numbering.value,
            hostname=hostname,
            inference_backend=self.name,
            inference_backend_properties=self.get_backend_properties(),
            metrics=metrics,
            start_time=Timer.get_instance("test_case").start_time,
            end_time=Timer.get_instance("test_case").end_time,
            execution_seconds=Timer.get_instance("test_case").elapsed_time,
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
                    filled_prompt = string.Template(unfilled_prompt).substitute(previous_completion_texts)
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
                message_history.add_assistant_message(prompt_step.prefix + message if prompt_step.prefix else message)

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

    def load_model_from_config(self, model_config: ModelConfig):
        if self.loaded_model:
            del self.loaded_model

        if not isinstance(model_config, HFModelConfig):
            raise ValueError("Only HF Models are supported by this inference backend")

        model_config: HFModelConfig

        InferenceBackend.ensure_hf_model_is_downloaded(local_path=self.model_folder_path, hf_repo=model_config.hf_repo, model_filename=model_config.hf_filename)

        # Load correct Tokenizer
        if model_config.use_hf_tokenizer:
            tokenizer = LlamaHFTokenizer.from_pretrained(model_config.hf_tokenizer)
            print(f"External Tokenizer {model_config.hf_tokenizer} loaded.")
        else:
            tokenizer = None  # Defaults to LlamaTokenizer

        Timer.get_instance(f"Load {model_config.hf_filename}").start_over()
        self.loaded_model: Llama = Llama(
            model_path=str(self.model_folder_path/model_config.hf_filename),
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            logits_all=self.logits_all,
            tokenizer=tokenizer,
            verbose=self.verbose
        )
        Timer.get_instance(f"Load {model_config.hf_filename}").end()

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
            prompt_tokens=completion_response["usage"]["prompt_tokens"],
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
    n_batch: int = 2048
    n_parallel: int

    process: Popen = None
    completion_url = "http://localhost:8080/completion"
    headers = {'content-type': 'application/json'}

    def __init__(self):
        # TODO: implement ensure_exists() function or python config file
        try:
            model_folder_path_str = os.environ.get("TB_MODEL_PATH")
            n_parallel = os.environ.get("TB_LLAMA_CPP_SERVER_SLOTS")
            if not model_folder_path_str or not n_parallel:
                raise KeyError
            else:
                self.model_folder_path: Path = Path(model_folder_path_str)
                self.model_folder_path.mkdir(parents=True, exist_ok=True)
                self.n_parallel = int(n_parallel)
        except KeyError:
            print("Please specify a model path. Did you forget to source .env?")
            exit()

        self.session: Session = Session()

    def load_model_from_config(self, model_config: ModelConfig):
        import subprocess

        if self.process:
            self.process.kill()

        if not isinstance(model_config, HFModelConfig):
            raise ValueError("Only HF Models are supported by this inference backend")

        model_config: HFModelConfig

        InferenceBackend.ensure_hf_model_is_downloaded(local_path=self.model_folder_path, hf_repo=model_config.hf_repo, model_filename=model_config.hf_filename)

        Timer.get_instance(f"Load {model_config.hf_filename}").start_over()

        self.process = subprocess.Popen([
            "../../llama.cpp/server",
            "-m", str(self.model_folder_path/model_config.hf_filename),
            "-b", str(self.n_batch),
            "-c", str(self.n_ctx),
            "-ngl", str(self.n_gpu_layers),
            "-np", str(self.n_parallel),
            "-cb",
            "--log-disable"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        time.sleep(3)
        Timer.get_instance(f"Load {model_config.hf_filename}").end()

        self.current_model_config = model_config

    def _run_test_case_subset(self, test_case: TestCase, thread_id: int, test_data_subset: List[SingleDataInstance], output_queue: Queue):
        print(f"Thread {thread_id} starting...")

        progressbar = tqdm(test_data_subset)
        progressbar.set_description(f"Benchmarking model on Thread {thread_id}")

        for single_test_data_instance in progressbar:
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

    def _run_test_case(self, test_case: TestCase, test_data_instances: List[SingleDataInstance]) -> List[SingleBenchmarkResult]:
        threads = []
        output_queue = Queue()

        # Split data into chunks for each process
        chunk_size = len(test_data_instances) // self.n_parallel
        chunks = [test_data_instances[i * chunk_size:(i + 1) * chunk_size] for i in range(self.n_parallel)]

        # Handle any remaining instances in the last chunk
        if len(test_data_instances) % self.n_parallel:
            chunks[-1].extend(test_data_instances[self.n_parallel * chunk_size:])

        print(f"{chunk_size=} {[len(chunk) for chunk in chunks]=}")

        for i in range(self.n_parallel):
            thread = threading.Thread(target=self._run_test_case_subset, args=(test_case, i, chunks[i], output_queue))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        all_results: List[SingleBenchmarkResult] = []
        while not output_queue.empty():
            all_results.append(output_queue.get())

        all_results.sort(key=lambda single_result: single_result.question_id)

        return all_results

    def create_completion(self, prompt: str, completion_config: CompletionConfig, decoder: Decoder, additional_params: Dict[str, Any]) -> CompletionResult:
        grammar_string = ""

        if type(decoder) == GreedyConstrainedDecoder:
            decoder: GreedyConstrainedDecoder
            grammar_string = self.get_grammar_string_from_labels(decoder.allowed_strings)

        if type(decoder) == GreedyDecoder:  # TODO
            pass

        if type(decoder) == BeamSearch:  # TODO
            pass

        request = {
            "prompt": prompt,
            "id_slot": additional_params["id_slot"],  # ensure that a thread only uses its own server slot
            "n_predict": completion_config.max_tokens,
            "n_probs": completion_config.max_logprobs,
            "temperature": completion_config.temperature,
            "samplers": ["temperature"],
            "repeat_last_n": 0,
            "min_p": 0.0,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "mirostat_eta": 0.0,
            "mirostat_tau": 0.0,
            "grammar": grammar_string
        }

        completion_response = self.session.post(url=self.completion_url, headers=self.headers, json=request).json()

        return self.__convert_completion_response(prompt, completion_response)

    @staticmethod
    def __convert_completion_response(prompt: str, completion_response: Dict) -> CompletionResult:
        def get_logprob(prob: float) -> float:
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
        # TODO: Implement
        return {}
