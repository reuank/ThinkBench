import datetime
import time
import os
import json
import requests
import asyncio
import aiohttp
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
from numpy import float32
from llama_cpp import Llama, LlamaGrammar, CreateCompletionResponse
import datasets
from huggingface_hub import hf_hub_download
from typing import List, Dict, Optional
from typing_extensions import TypedDict
from tqdm import tqdm

llama_instruct_template = {
    "begin_question": "<s>[INST]",
    "model_handoff": "[/INST]"
}

models = {
    "orca-2-7b": { # pip3 install sentencepiece protobuf
        "hf-repo": "TheBloke/Orca-2-7B-GGUF",
        "hf-tokenizer": "microsoft/Orca-2-7b",
        "template": {
            "begin_question": "<|im_start|>user\n",
            "model_handoff": "'<|im_end|>\n<|im_start|>assistant\n"
        },
        "use_template": True
    },
    "orca-2-13b": {
        "hf-repo": "TheBloke/Orca-2-13B-GGUF",
        "hf-tokenizer": "microsoft/Orca-2-7b",
        "template": {
            "begin_question": "<|im_start|>user\n",
            "model_handoff": "'<|im_end|>\n<|im_start|>assistant\n"
        },
        "use_template": True
    },
    "llama-2-7b-chat": {
        "hf-repo": "TheBloke/Llama-2-7B-Chat-GGUF",
        "template": llama_instruct_template,
        "use_template": False
    },
    "llama-2-13b-chat": {
        "hf-repo": "TheBloke/Llama-2-13B-Chat-GGUF",
        "template": llama_instruct_template,
        "use_template": False
    },
    "phi-2": {
        "hf-repo": "TheBloke/Phi-2-GGUF",
        "hf-tokenizer": "microsoft/phi-2",
        "template": {
            "begin_question": "Instruct:",
            "model_handoff": "Output:"
        },
        "use_template": True
    },
    "mistral-7b-instruct-v0.2": {
        "hf-repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        #"hf-tokenizer": "mistralai/Mistral-7B-Instruct-v0.2",
        "template": llama_instruct_template,
        "use_template": True
    }
}

model_folder_path = "/Users/leonknauer/code/uni/thesis/models"  #"/home/kit/itas/ep8668/models"

class SingleResult(TypedDict):
    question_id: int
    question: str
    answers: List[str]
    labels: List[str]
    prompt: str
    model_choice: str
    correct_answer: str
    is_correct: bool
    top_logprobs: Dict[str, float]


class TestResult(TypedDict):
    model: str
    start_time: float
    end_time: float
    execution_seconds: float
    total_accuracy: float
    prompt_template: str
    comment: str
    results: List[SingleResult]
    server_properties: Optional[Dict]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float32):
            return float(obj)


def test_baseline_in_process(max_questions: int = -1, log_result: bool = True, comment: str = "", template: Dict = None):
    start_time = time.time()

    correct_counter = 0
    results = []
    num_questions = len(dataset['id']) if max_questions == -1 else max_questions

    for question_id, question in enumerate(tqdm(dataset['question'][:num_questions])):
        if max_questions != -1 and question_id >= max_questions:
            break

        choices = dataset['choices'][question_id]
        labels = choices['label']  # [A, B, C, D]
        answers = choices['text']
        correct_answer = dataset['answerKey'][question_id]

        prompt = non_cot_decision_prompt(question, labels, answers, template)

        response: CreateCompletionResponse = model.create_completion(
            prompt=prompt,
            temperature=0,
            logprobs=10,
            grammar=get_llama_grammar_from_labels(labels)
        )

        model_choice = response["choices"][0]["text"]

        if is_equal(model_choice, correct_answer):
            correct_counter += 1

        results.append(
            SingleResult(
                question_id=question_id,
                question=question,
                answers=answers,
                labels=labels,
                prompt=prompt,
                model_choice=model_choice,
                correct_answer=correct_answer,
                is_correct=is_equal(model_choice, correct_answer),
                top_logprobs=response["choices"][0]["logprobs"]["top_logprobs"][0]
            )
        )

    end_time = time.time()

    total_accuracy = round(correct_counter * 100/num_questions, 2)
    test_result = TestResult(
        model=model_filename,
        start_time=start_time,
        end_time=end_time,
        execution_seconds=round(time.time() - start_time, 2),
        total_accuracy=total_accuracy,
        prompt_template=non_cot_decision_prompt("[Q]", ["[Label1]", "[Label2]"],  ["[Answer1]", "[Answer2]"], template),
        comment=comment,
        results=results,
    )

    print(f"Execution took {round(time.time() - start_time, 2)} seconds.")
    print(f"Total accuracy: {total_accuracy} %.")

    if log_result:
        filename = f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_baseline_arc_test_{num_questions}_{model_filename}_in_process.json"
        f = open(filename, "a")
        f.write(json.dumps(test_result, cls=NumpyEncoder, indent=4))
        f.close()
        print(f"File {filename} written.")


def test_baseline_on_server_async(max_questions: int = -1, log_result: bool = True, endpoint="http://localhost:8080", comment: str = ""):
    start_time = time.time()

    num_questions = len(dataset['id']) if max_questions == -1 else max_questions

    async def get_single(question_id, session, max_retries=10):
        question = dataset['question'][question_id]
        choices = dataset['choices'][question_id]
        labels = choices['label']  # [A, B, C, D]
        answers = choices['text']
        correct_answer = dataset['answerKey'][question_id]

        prompt = non_cot_decision_prompt(question, labels, answers)

        url = f"{endpoint}/completion"
        headers = {'content-type': 'application/json'}
        data = {
            "prompt": prompt,
            "temperature": 0,
            "n_predict": 1,
            "n_probs": 10,
            "grammar": get_grammar_string_from_labels(labels)
        }

        retries = 0
        while retries < max_retries:
            try:
                async with session.post(url=url, headers=headers, json=data, raise_for_status=True) as response:
                    response_json = await response.json()

                    model_choice = response_json["content"]

                    return SingleResult(
                            question_id=question_id,
                            question=question,
                            answers=answers,
                            labels=labels,
                            prompt=prompt,
                            model_choice=model_choice,
                            correct_answer=correct_answer,
                            is_correct=is_equal(model_choice, correct_answer),
                            top_logprobs=response_json["completion_probabilities"]
                        )

            except aiohttp.ClientError as e:  # Catch HTTP errors and possibly other specific exceptions you expect
                print(f"Attempt number {retries + 1} failed for question_id {question_id}: {e}")
                retries += 1


            except Exception as e:  # Catch other exceptions, you might want to handle these differently
                print(f"Unexpected error for question_id {question_id}, not retrying: {e}")
                break

    async def get_all():
        async with aiohttp.ClientSession() as session:
            async_results = await asyncio.gather(
                *(get_single(question_id, session) for question_id in range(num_questions))
            )

        return async_results

    results = asyncio.run(get_all())

    end_time = time.time()

    server_properties = requests.get(url=f"{endpoint}/props", headers={'content-type': 'application/json'}).json()
    model_filename = os.path.basename(os.path.normpath(server_properties["default_generation_settings"]["model"]))

    num_correct = sum(map(lambda x: 1, filter(lambda result: result["is_correct"], results)))
    total_accuracy = round((num_correct / num_questions) * 100, 2)

    test_result = TestResult(
        model=model_filename,
        start_time=start_time,
        end_time=end_time,
        execution_seconds=round(time.time() - start_time, 2),
        total_accuracy=total_accuracy,
        prompt_template=non_cot_decision_prompt("[Q]", ["[Label1]", "[Label2]"],  ["[Answer1]", "[Answer2]"]),
        comment=comment,
        results=results,
        server_properties=server_properties
    )

    print(f"Execution took {round(time.time() - start_time, 2)} seconds.")
    print(f"Total accuracy: {total_accuracy} %.")

    if log_result:
        filename = f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_baseline_arc_test_{num_questions}_{model_filename}_on_server_async.json"
        f = open(filename, "a")
        f.write(json.dumps(test_result, cls=NumpyEncoder, indent=4))
        f.close()
        print(f"File {filename} written.")


def test_baseline_on_server_sequential(max_questions: int = -1, log_result: bool = True, endpoint="http://localhost:8080", comment: str = ""):
    global model_filename

    start_time = time.time()

    correct_counter = 0
    results = []
    num_questions = len(dataset['id']) if max_questions == -1 else max_questions

    for question_id, question in enumerate(tqdm(dataset['question'][:num_questions])):
        choices = dataset['choices'][question_id]
        labels = choices['label']  # [A, B, C, D]
        answers = choices['text']
        correct_answer = dataset['answerKey'][question_id]

        prompt = non_cot_decision_prompt(question, labels, answers)

        url = f"{endpoint}/completion"
        headers = {'content-type': 'application/json'}
        data = {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": 10,
            "grammar": get_grammar_string_from_labels(labels)
        }

        response = requests.post(url, data=json.dumps(data), headers=headers).json()

        model_choice = response["content"]

        if is_equal(model_choice, correct_answer):
            correct_counter += 1

        results.append(
            SingleResult(
                question_id=question_id,
                question=question,
                answers=answers,
                labels=labels,
                prompt=prompt,
                model_choice=model_choice,
                correct_answer=correct_answer,
                is_correct=is_equal(model_choice, correct_answer),
                top_logprobs=response["completion_probabilities"]
            )
        )

    end_time = time.time()

    server_properties = requests.get(url=f"{endpoint}/props", headers={'content-type': 'application/json'}).json()
    model_filename = os.path.basename(os.path.normpath(server_properties["default_generation_settings"]["model"]))

    total_accuracy = round(correct_counter * 100 / num_questions, 2)
    test_result = TestResult(
        model=model_filename,
        start_time=start_time,
        end_time=end_time,
        execution_seconds=round(time.time() - start_time, 2),
        total_accuracy=total_accuracy,
        prompt_template=non_cot_decision_prompt("[Q]", ["[Label1]", "[Label2]"],  ["[Answer1]", "[Answer2]"]),
        comment=comment,
        results=results,
        server_properties=server_properties
    )

    print(f"Execution took {round(time.time() - start_time, 2)} seconds.")
    print(f"Total accuracy: {total_accuracy} %.")

    if log_result:
        filename = f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_baseline_arc_test_{num_questions}_{model_filename}_on_server_sequential.json"
        f = open(filename, "a")
        f.write(json.dumps(test_result, cls=NumpyEncoder, indent=4))
        f.close()
        print(f"File {filename} written.")


def is_equal(answer: str, reference: str):
    # TODO: define different rules for equlity

    return answer.strip() == reference


def get_llama_grammar_from_labels(labels: [str]) -> LlamaGrammar:
    return LlamaGrammar.from_string(
        grammar=get_grammar_string_from_labels(labels),
        verbose=False
    )

def get_grammar_string_from_labels(labels: [str]) -> str:
    labels_with_quotes = list(map(lambda label: f'"{label}"', labels))

    return f"root   ::= [ ]? option \noption ::= ({'|'.join(labels_with_quotes) })"


def non_cot_decision_prompt(question: str, labels: [str], answers: [str], template: Dict) -> str: # TODO: Make prompt model specific
    options = ""

    for label_id, label in enumerate(labels):
        options += f"({label}) {answers[label_id]}\n"

    if template is None:
        template = {"begin_question": "", "model_handoff": ""}

    return f"{template['begin_question']} Question: {question} \n" \
           f"Answer Choices: {options}" \
           f"Among {labels[0]} through {labels[-1]}, the correct answer is: {template['model_handoff']} "\
           # f"Among {labels[0]} through {labels[-1]}, what is the correct answer?{template['model_handoff']} "


def run_all_baselines(max_questions: int = -1, comment: str = ""):
    global model, model_filename  #Todo: not use global

    start_tests = time.time()

    for model_name in models:
        print("=" * 70)
        load_model(model_name)
        test_baseline_in_process(
            max_questions=max_questions,
            log_result=True,
            comment=comment,
            template=models[model_name]["template"] if models[model_name]["use_template"] else None
        )

    print("=" * 70)
    end_tests = timqe.time()
    print(f"All tests took {int((end_tests - start_tests) / 60)} minutes {int((end_tests - start_tests) % 60)} seconds.")
    print("=" * 70)

def run_single_baseline_in_process(model_name: str, max_questions=1, log_result=True):
    load_model(model_name)
    test_baseline_in_process(max_questions=max_questions, log_result=log_result)


def run_single_baseline_on_server(max_questions=1, log_result=True):
    # TODO: Instantiate Backend
    test_baseline_on_server_async(max_questions=max_questions, log_result=log_result)


def load_model(model_name: str):
    global model, model_filename  # Todo: not use global

    model_filename = f"{model_name}.Q4_K_M.gguf"
    model_path = f"{model_folder_path}/{model_filename}"

    if not os.path.isdir(model_folder_path):
        os.mkdir(model_folder_path)
        print("Model folder created, as it does not yet exist.")

    if not os.path.isfile(model_path):
        hf_hub_download(
            repo_id=models[model_name]["hf-repo"],
            filename=model_filename,
            local_dir=model_folder_path,
            local_dir_use_symlinks=False
        )
        print(f"Model {model_filename} downloaded, as it does not yet exist.")

    # Load correct Tokenizer
    if "hf-tokenizer" in models[model_name].keys():
        tokenizer = LlamaHFTokenizer.from_pretrained(models[model_name]["hf-tokenizer"])
        print(f"External Tokenizer {models[model_name]['hf-tokenizer']} loaded.")
    else:
        tokenizer = None  # Defaults to LlamaTokenizer

    start_model_load = time.time()
    model = Llama(
        model_path=model_path,
        n_gpu_layers=1000,
        n_ctx=8192,
        n_batch=1024,
        logits_all=True,
        tokenizer=tokenizer,
        verbose=False
    )
    end_model_load = time.time()

    print(f"Model {model_filename} loaded in {round(end_model_load - start_model_load, 2)} seconds.")


if __name__ == '__main__':
    start_dataset_load = time.time()

    datasets.logging.set_verbosity_info()
    dataset = datasets.load_dataset(
        path="ai2_arc",
        name="ARC-Challenge",
        split="test"
        # num_proc=8,
        # keep_in_memory=True
    )
    end_dataset_load = time.time()

    # TODO: Improve dataset load time (why faster in first draft?)
    print(f"Dataset loaded in {round(end_dataset_load - start_dataset_load, 2)} seconds")

    model: Llama
    model_filename: str

    run_all_baselines(max_questions=-1, comment="All baselines with partially activated chat template, standard non cot prompt")
    #run_single_baseline_in_process(model_name="llama-2-7b-chat", max_questions=100, log_result=True) # TODO: Add test run comment
    #run_baseline_on_server(max_questions=100, log_result=True) # TODO: Add test run comment

