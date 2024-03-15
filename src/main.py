import datetime
import time
import os
import json
import requests
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
from numpy import float32
from llama_cpp import Llama, LlamaGrammar, CreateCompletionResponse
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from typing import List, Dict
from typing_extensions import TypedDict
from tqdm import tqdm

models = {
    "llama-2-7b-chat": {
        "hf-repo": "TheBloke/Llama-2-7B-Chat-GGUF"
    },
    "llama-2-13b-chat": {
        "hf-repo": "TheBloke/Llama-2-13B-Chat-GGUF"
    },
    "phi-2": {
        "hf-repo": "TheBloke/Phi-2-GGUF",
        "hf-tokenizer": "microsoft/phi-2",
        "keywords": { # TODO: Implement Keywords
            "begin_question": "Instruct:",
            "model_handoff": "Output:"
        }
    },
    "mistral-7b-instruct-v0.2": {
        "hf-repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "hf-tokenizer": "mistralai/Mistral-7B-Instruct-v0.2"
    }
}

model_folder_path = "/Users/leonknauer/code/uni/thesis/models/TheBloke/Llama-2-7B-Chat-GGUF/" #"/home/kit/itas/ep8668/models"

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
    results: List[SingleResult]
    accuracies: List[float]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float32):
            return float(obj)


def test_baseline_in_process(max_questions: int = -1, log_result: bool = True):
    start_time = time.time()

    correct_counter = 0
    current_accuracy = 0
    results = []
    accuracies = []
    num_questions = len(dataset['id']) if max_questions == -1 else max_questions

    for question_id, question in enumerate(tqdm(dataset['question'][:num_questions])):
        if max_questions != -1 and question_id >= max_questions:
            break

        choices = dataset['choices'][question_id]
        labels = choices['label']  # [A, B, C, D]
        answers = choices['text']
        correct_answer = dataset['answerKey'][question_id]

        prompt = non_cot_decision_prompt(question, labels, answers)

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

        current_accuracy = round(correct_counter * 100/(question_id+1), 2)
        accuracies.append(current_accuracy)

    end_time = time.time()

    test_result = TestResult(
        model=model_filename,
        start_time=start_time,
        end_time=end_time,
        execution_seconds=round(time.time() - start_time, 2),
        total_accuracy=current_accuracy,
        prompt_template=non_cot_decision_prompt("[Q]", ["[Label1]", "[Label2]"],  ["[Answer1]", "[Answer2]"]),
        results=results,
        accuracies=accuracies
    )

    if log_result:
        f = open(f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_baseline_arc_test_{num_questions}_{model_filename}_in_process.json", "a")
        f.write(json.dumps(test_result, cls=NumpyEncoder, indent=4))
        f.close()

    print(f"Execution took {round(time.time() - start_time, 2)} seconds.")
    print(f"Total accuracy: {current_accuracy} %.")


def test_baseline_on_server(max_questions: int = -1, log_result: bool = True):
    global model_filename

    start_time = time.time()

    correct_counter = 0
    current_accuracy = 0
    results = []
    accuracies = []
    num_questions = len(dataset['id']) if max_questions == -1 else max_questions

    for question_id, question in enumerate(tqdm(dataset['question'][:num_questions])):
        choices = dataset['choices'][question_id]
        labels = choices['label']  # [A, B, C, D]
        answers = choices['text']
        correct_answer = dataset['answerKey'][question_id]

        prompt = non_cot_decision_prompt(question, labels, answers)

        url = "http://localhost:8080/completion"
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

        current_accuracy = round(correct_counter * 100/(question_id+1), 2)
        accuracies.append(current_accuracy)

    end_time = time.time()

    model_path = response["generation_settings"]["model"]
    model_filename = os.path.basename(os.path.normpath(model_path))

    test_result = TestResult(
        model=model_filename,
        start_time=start_time,
        end_time=end_time,
        execution_seconds=round(time.time() - start_time, 2),
        total_accuracy=current_accuracy,
        prompt_template=non_cot_decision_prompt("[Q]", ["[Label1]", "[Label2]"],  ["[Answer1]", "[Answer2]"]),
        results=results,
        accuracies=accuracies
    )

    if log_result:
        f = open(f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_baseline_arc_test_{num_questions}_{model_filename}_on_server.json", "a")
        f.write(json.dumps(test_result, cls=NumpyEncoder, indent=4))
        f.close()

    print(f"Execution took {round(time.time() - start_time, 2)} seconds.")
    print(f"Total accuracy: {current_accuracy} %.")


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


def non_cot_decision_prompt(question: str, labels: [str], answers: [str]) -> str: # TODO: Make prompt model specific
    options = ""

    for label_id, label in enumerate(labels):
        options += f"({label}) {answers[label_id]}\n"

    return f"Question: {question} \n" \
           f"Answer Choices: {options}" \
           f"Among {labels[0]} through {labels[-1]}, the answer is: "


def run_all_baselines():
    global model, model_filename #Todo: not use global

    for model_name in models:
        load_model(model_name)
        test_baseline_in_process(max_questions=-1, log_result=True)


def run_single_baseline_in_process(model_name: str, max_questions=1, log_result=True):
    load_model(model_name)
    test_baseline_in_process(max_questions=max_questions, log_result=log_result)


def run_single_baseline_on_server(max_questions=1, log_result=True):
    # TODO: Instantiate Backend
    test_baseline_on_server(max_questions=max_questions, log_result=log_result)


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

    print(f"Model {model_filename} loaded in {end_model_load - start_model_load} seconds.")


if __name__ == '__main__':
    start_dataset_load = time.time()
    dataset = load_dataset(
        path="ai2_arc",
        name="ARC-Challenge",
        split="test"
        # num_proc=8,
        # keep_in_memory=True
    )
    end_dataset_load = time.time()

    # TODO: Improve dataset load time (why faster in first draft?)
    print(f"Dataset loaded in {end_dataset_load - start_dataset_load} seconds")

    model: Llama
    model_filename: str

    #run_single_baseline_in_process(model_name="llama-2-7b-chat", max_questions=100, log_result=True) # TODO: Add test run comment
    run_single_baseline_on_server(max_questions=100, log_result=True) # TODO: Add test run comment

