import os
import signal
import sys
import threading
import socket
import time
from pathlib import Path
from queue import Queue
from string import Template
from typing import List, Dict, Any

import git
import psutil
from requests import Session
from tqdm import tqdm

from benchmark.benchmark import SingleBenchmarkResult
from constants import COMPLETION_SEED, N_GPU_LAYERS, INFERENCE_BACKEND_VERBOSE, TOKENIZE_BEFORE, SERVER_HOST, \
    DEFAULT_MODEL_PATH
from dataset.single_data_instance import SingleDataInstance
from inference.completion import CompletionConfig, CompletionResult, Choice, Logprobs, Usage
from inference.decoder import Decoder, GreedyConstrainedDecoder, TemperatureDecoder, GreedyDecoder, NucleusDecoder
from inference.inference_backend import InferenceBackend, INFERENCE_BACKEND_REGISTRY
from benchmark.testcase import TestCase
from model_config.hf_model_config import HFModelConfig
from model_config.model_config import ModelConfig, QuantizationMethod
from utils.env_loader import EnvReader
from utils.logger import Logger
from utils.timer import Timer


@INFERENCE_BACKEND_REGISTRY.register(name="llama.cpp")
class LlamaCppServerInferenceBackend(InferenceBackend):
    n_ctx: int = 8192
    n_batch: int = 4096
    n_parallel: int = 1

    process: psutil.Popen = None
    completion_url_template: Template = Template("http://${host}:${port}/completion")
    tokenization_url_template: Template = Template("http://${host}:${port}/tokenize")
    properties_url_template: Template = Template("http://${host}:${port}/props")
    headers = {'content-type': 'application/json'}

    def __init__(self):
        self.model_folder_path: Path = Path(EnvReader.get("TB_MODEL_PATH", DEFAULT_MODEL_PATH))
        self.model_folder_path.mkdir(parents=True, exist_ok=True)
        Logger.info(f"Using model path {self.model_folder_path}")

        self.n_parallel: int = int(EnvReader.get("TB_LLAMA_CPP_SERVER_SLOTS", required=True))
        self.server_binary_path: Path = Path(EnvReader.get("LLAMA_CPP_SERVER_BINARY", required=True))
        self.port: int = int(EnvReader.get("LLAMA_CPP_SERVER_PORT", "8080"))

        self.n_ctx = self.n_parallel * 4096
        self.session: Session = Session()

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

    def load_model_from_config(self, model_config: ModelConfig):
        import subprocess

        Logger.info("Terminating any old server processes...")
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

        def is_port_open(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((SERVER_HOST, port))
                return result == 0

        if is_port_open(self.port):
            raise ValueError(f"Server could not be started, port {self.port} is already in use.")

        server_process_arguments = [
            str(self.server_binary_path),
            "--port", str(self.port),
            "-m", str(self.model_folder_path/hf_filename),
            "-b", str(self.n_batch),
            "-c", str(self.n_ctx),
            "-ngl", str(N_GPU_LAYERS),
            "-np", str(self.n_parallel),
            "--log-disable"
        ]

        if INFERENCE_BACKEND_VERBOSE:
            stdout = sys.stdout
        else:
            stdout = subprocess.DEVNULL

        self.process = subprocess.Popen(server_process_arguments, stdout=stdout, stderr=subprocess.STDOUT)
        time.sleep(2)
        Logger.info(f"Currently loaded model on the available server: {self.get_backend_properties()['loaded_model']}")
        Timer.get_instance(f"Load {hf_filename}").end(print_timer=True)

        self.current_model_config = model_config

    @staticmethod
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

            return chunks

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

        chunks = self.distribute_chunks(data=test_data_instances, num_threads=self.n_parallel)
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
        if type(decoder) == GreedyConstrainedDecoder or type(decoder) == GreedyDecoder:
            decoder.temperature = -1.0  # Return probs even when using greedy decoding
            samplers = ["temperature"]
        elif type(decoder) == TemperatureDecoder:
            samplers = ["temperature"]
        elif type(decoder) == NucleusDecoder:
            samplers = ["top_p"]

        if TOKENIZE_BEFORE:
            prompt = self.tokenize(prompt)

        request = self.construct_request(
            prompt=prompt,
            completion_config=completion_config,
            samplers=samplers,
            decoder=decoder,
            additional_params=additional_params
        )

        raw_completion_response = self.session.post(
            url=self.completion_url_template.substitute(port=self.port, host=SERVER_HOST),
            headers=self.headers,
            json=request
        ).json()

        completion_response = self.__convert_completion_response(prompt, raw_completion_response)

        if type(decoder) == GreedyConstrainedDecoder:
            # Using a grammar directly in llama.cpp does seem to work differently as compared to llama-cpp-python.
            decoder: GreedyConstrainedDecoder

            allowed_tokens = []
            for allowed_token in decoder.allowed_strings:
                allowed_tokens.append(f"{allowed_token}")
                allowed_tokens.append(f" {allowed_token}")

            if not completion_response.choices[0].logprobs.top_logprobs:
                completion_response.choices[0].logprobs.top_logprobs.append({"NONE": 1.0})

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

    def tokenize(self, prompt: str) -> List[int]:
        return self.session.post(
            url=self.tokenization_url_template.substitute(port=self.port, host=SERVER_HOST),
            headers=self.headers,
            json={
                "content": prompt,
                "add_special": False
            }
        ).json()["tokens"]

    @staticmethod
    def construct_request(
            prompt: str,
            completion_config: CompletionConfig,
            samplers: List[str],
            additional_params: Dict[str, Any],
            decoder: Decoder = GreedyDecoder()
    ) -> Dict[int, Any]:
        return {
            "prompt": prompt,
            "id_slot": additional_params["id_slot"] if "id_slot" in additional_params.keys() else -1,
            "n_predict": completion_config.max_tokens,
            "n_probs": completion_config.max_logprobs,
            "min_keep": completion_config.max_logprobs,
            "cache_prompt": completion_config.cache_prompt,
            "samplers": samplers,
            "seed": COMPLETION_SEED,
            "temperature": decoder.temperature,
            "repeat_penalty": decoder.repeat_penalty,
            "repeat_last_n": decoder.repeat_last_n,
            "min_p": decoder.min_p,
            "top_p": decoder.top_p,
            "top_k": decoder.top_k,
            "mirostat_eta": 0.0,
            "mirostat_tau": 0.0,
        }

    @staticmethod
    def __convert_completion_response(prompt: str, completion_response: Dict) -> CompletionResult:
        def get_logprob(prob: float) -> float:
            return prob
            # return math.log(prob) if prob != 0 else -100.0

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
        server_settings = self.session.get(
            url=self.properties_url_template.substitute(port=self.port, host=SERVER_HOST),
            headers=self.headers
        ).json()

        return {
            "loaded_model": server_settings["default_generation_settings"]["model"],
            "n_slots": server_settings["total_slots"],
            "n_ctx_per_slot": server_settings["default_generation_settings"]["n_ctx"],
            "batch_size": self.n_batch,
            "n_gpu_layers": N_GPU_LAYERS,
            "current_commit_hash": self.current_commit_hash
        }

    def terminate_all_running_servers(self):
        program_name = str(self.server_binary_path)

        """ Kills all processes that contain the program_name in their executable path. """
        for proc in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                # Check if process name or the executable matches the program name
                if program_name in proc.info['name'] or (proc.info['exe'] and program_name in proc.info['exe']):
                    Logger.info(f"Killing process '{proc.info['name']}' with PID {proc.info['pid']}")
                    proc.send_signal(signal.SIGTERM)  # or proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass  # Process has been killed or can't be accessed
