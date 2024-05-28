import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from string import Template
from typing import List, Dict, Any, Optional

import git
import psutil
from requests import Session
from tqdm import tqdm

from benchmark.benchmark import SingleBenchmarkResult
from constants import COMPLETION_SEED, N_GPU_LAYERS, INFERENCE_BACKEND_VERBOSE, TOKENIZE_BEFORE
from dataset.single_data_instance import SingleDataInstance
from inference.completion import CompletionConfig, CompletionResult, Choice, Logprobs, Usage
from inference.decoder import Decoder, GreedyConstrainedDecoder, TemperatureDecoder, GreedyDecoder, BeamSearchDecoder, \
    Beam
from inference.inference_backend import InferenceBackend, INFERENCE_BACKEND_REGISTRY
from benchmark.testcase import TestCase
from model_config.hf_model_config import HFModelConfig
from model_config.model_config import ModelConfig, QuantizationMethod
from utils.encoders import TotalResultEncoder
from utils.timer import Timer


@INFERENCE_BACKEND_REGISTRY.register("llama.cpp")
class LlamaCppServerInferenceBackend(InferenceBackend):
    n_ctx: int = 8192
    n_batch: int = 4096
    n_parallel: int = 1

    process: psutil.Popen = None
    completion_url_template: Template = Template("http://localhost:${port}/completion")
    tokenization_url_template: Template = Template("http://localhost:${port}/tokenize")
    properties_url_template: Template = Template("http://localhost:${port}/props")
    headers = {'content-type': 'application/json'}

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
            print("Please specify a model path, the number of server slots and the server binary path. "
                  "Did you forget to source .env?")
            exit()

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
        print(f"Currently loaded model on the available server: {self.get_backend_properties()['loaded_model']}")
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
        elif type(decoder) == BeamSearchDecoder:
            decoder: BeamSearchDecoder
            return self.beam_search(prompt, completion_config, decoder, additional_params)

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
            url=self.completion_url_template.substitute(port=self.port),
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
            url=self.tokenization_url_template.substitute(port=self.port),
            headers=self.headers,
            json={
                "content": prompt,
                "add_special": False
            }
        ).json()["tokens"]

    def generate_single_token(
            self,
            prompt: str,
            additional_params: Dict[str, Any],
            decoder: Decoder = GreedyDecoder()
    ) -> CompletionResult:
        if TOKENIZE_BEFORE:
            prompt = self.tokenize(prompt)

        if type(decoder) == GreedyDecoder:
            decoder.temperature = -1.0

        request = self.construct_request(
            prompt=prompt,
            completion_config=CompletionConfig(max_tokens=1, max_logprobs=100),
            samplers=["temperature"],
            decoder=decoder,
            additional_params=additional_params
        )

        raw_completion_response = self.session.post(
            url=self.completion_url_template.substitute(port=self.port),
            headers=self.headers,
            json=request
        ).json()

        return self.__convert_completion_response(prompt, raw_completion_response)

    def beam_search(
            self,
            prompt: str,
            completion_config: CompletionConfig,
            decoder: BeamSearchDecoder,
            additional_params: Dict[str, Any]
    ) -> CompletionResult:
        eos_token = "</s>"
        beams = [Beam(prompt=prompt, generated_tokens=[], log_prob_sum=0.0, log_probs=[])]
        completed_beams = []

        completion_count = 0

        log_beam_search = False

        for token_id in range(completion_config.max_tokens):
            if log_beam_search: print(f"Token ID: {token_id}")
            all_candidates = []
            for beam_id, beam in enumerate(beams):
                completion_response: CompletionResult = self.create_completion(
                    prompt=beam.get_current_prompt(),
                    completion_config=CompletionConfig(
                        max_tokens=1,
                        max_logprobs=decoder.beam_width,
                        cache_prompt=False
                    ),
                    decoder=GreedyDecoder(),
                    additional_params=additional_params
                )

                completion_count += 1

                top_tokens: Dict[str, float] = completion_response.get_last_token_logprobs()
                if log_beam_search:
                    print(f"Beam {beam_id} generated tokens: {beam.generated_tokens}")
                    print(f"Beam {beam_id} top tokens: {top_tokens}")
                    print(f"Beam {beam_id} logprob sum: {beam.log_prob_sum}")

                def get_logprob(prob: float) -> float:
                    return math.log(prob) if prob != 0 else -100.0

                for token, prob in top_tokens.items():
                    new_beam = Beam(
                        prompt=prompt,
                        generated_tokens=beam.generated_tokens + [token],
                        log_prob_sum=beam.log_prob_sum + get_logprob(prob),
                        log_probs=beam.log_probs + [prob]
                    )
                    all_candidates.append(new_beam)

                if log_beam_search:
                    print(f"Beam {beam_id} all candidates: {[''.join(candidate.generated_tokens) for candidate in all_candidates]}")

                if log_beam_search and beam_id != len(beams) - 1:
                    print("")

            beams = sorted(all_candidates, key=lambda x: x.get_beam_search_score(), reverse=True)

            for beam_id in range(decoder.beam_width):
                beam = beams[beam_id]
                if beam.generated_tokens and beam.generated_tokens[-1] == eos_token:
                    completed_beams.append(beam)
                    if log_beam_search:
                        print(f"Beam {beam_id} finished with completion '{beam.get_completion()}'")
                    beams.pop(beam_id)

            if log_beam_search:
                print("="*40)

            beams = beams[:decoder.beam_width]

            if all((beam.generated_tokens and beam.generated_tokens[-1] == eos_token) for beam in beams):
                completed_beams.extend([beam for beam in beams if beam.generated_tokens[-1] == eos_token])
                beams = [beam for beam in beams if beam.generated_tokens[-1] != eos_token]
                if log_beam_search:
                    print("All beams finished!")
                break

        if completed_beams:
            print(f"Completed beams: \n {json.dumps([''.join(completed_beam.generated_tokens) + f'({completed_beam.get_beam_search_score()})' for completed_beam in completed_beams], indent=2)}")
            best_beam = max(completed_beams, key=lambda x: x.get_beam_search_score())
        else:
            best_beam = max(beams, key=lambda x: x.get_beam_search_score())

        if log_beam_search:
            print(f"Number of completions: {completion_count}")

        return CompletionResult(
            id="beam",
            prompt=prompt,
            object="",
            created=int(time.time()),
            model=self.current_model_config.model_name,
            choices=[
                Choice(
                    text="".join(best_beam.generated_tokens),
                    logprobs=Logprobs(
                        tokens=best_beam.generated_tokens,
                        text_offset=[],
                        token_logprobs=[],
                        top_logprobs=[{k: v for k, v in zip(best_beam.generated_tokens, best_beam.log_probs)}]
                    ),
                    finish_reason="stopped_eos" if best_beam.generated_tokens and best_beam.generated_tokens[-1] == eos_token else "stopped_limit",
                    index=0
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt),
                prompt_tokens_per_second=0.0,
                prompt_ms=0.0,
                completion_tokens=0,
                completion_tokens_per_second=0.0,
                completion_ms=0.0,
                total_tokens=0
            )
        )

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
        server_settings = self.session.get(url=self.properties_url_template.substitute(port=self.port), headers=self.headers).json()

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
                    print(f"Killing process '{proc.info['name']}' with PID {proc.info['pid']}")
                    proc.send_signal(signal.SIGTERM)  # or proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass  # Process has been killed or can't be accessed
