import pathlib
import uuid

import git
import os
from string import Template
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Dict, Any

from inference.completion import CompletionResult, CompletionHistory, CompletionConfig
from inference.decoder import Decoder
from benchmark.results import SingleBenchmarkResult, TestCaseResult
from dataset.single_data_instance import SingleDataInstance
from inference.message_history import MessageHistory
from benchmark.prompt_chain import PromptChain, PromptCompletionStep, PromptTemplateStep, PromptTextStep
from benchmark.testcase import TestCase
from model_config.model_config import ModelConfig, QuantizationMethod
from utils.registry import Registry
from utils.timer import Timer


class InferenceBackend(ABC):
    current_model_config: ModelConfig
    loaded_model: None

    @property
    def name(self):
        return self.__class__.__name__

    @property
    @abstractmethod
    def supported_quantization_methods(self) -> List[QuantizationMethod]:
        raise NotImplementedError

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

    @abstractmethod
    def get_backend_properties(self):
        raise NotImplementedError


INFERENCE_BACKEND_REGISTRY = Registry(
    registry_name="inference_backends",
    base_class=InferenceBackend
)
