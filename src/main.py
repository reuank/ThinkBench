import datetime

from numpy import float32

from llama_cpp import Llama, LlamaGrammar, CreateCompletionResponse
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import time
import os
import json
from typing import Any, List, Optional, Dict, Union
from typing_extensions import TypedDict, NotRequired, Literal
from tqdm import tqdm

models = {
    "llama-2-7b-chat": {
        "hf-repo": "TheBloke/Llama-2-7B-Chat-GGUF"
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
    accuracies: List[float]
    results: List[SingleResult]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float32):
            return float(obj)


lmql_answers = ["C", "B", "B", "D", "B", "B", "C", "C", "A", "A", "C", "A", "B", "D", "D", "A", "C", "D", "B",
                        "D", "B", "B", "A", "C", "B", "B", "B", "B", "C", "B", "D", "C", "C", "C", "B", "B", "B", "D",
                        "C", "C", "C", "B", "C", "D", "B", "B", "C", "C", "B", "B", "D", "C", "B", "C", "B", "B", "D",
                        "A", "D", "B", "B", "B", "D", "C", "C", "A", "B", "D", "B", "A", "D", "C", "B", "D", "B", "C",
                        "B", "B", "B", "A", "D", "C", "B", "D", "C", "D", "A", "B", "B", "C", "A", "D", "C", "A", "B",
                        "B", "C", "A", "A", "B", "C", "C", "A", "C", "B", "A", "D", "B", "D", "D", "C", "A", "C", "A",
                        "B", "D", "C", "C", "B", "C", "A", "B", "B", "A", "A", "D", "B", "B", "B", "D", "D", "B", "C",
                        "B", "B", "B", "B", "B", "A", "A", "D", "C", "C", "B", "A", "C", "B", "C", "C", "A", "D", "D",
                        "B", "C", "B", "B", "B", "C", "B", "C", "D", "D", "C", "B", "A", "A", "B", "B", "D", "D", "A",
                        "D", "C", "C", "D", "B", "C", "D", "A", "C", "A", "D", "B", "C", "A", "A", "D", "B", "C", "D",
                        "B", "C", "A", "D", "B", "A", "B", "B", "B", "C", "A", "C", "A", "C", "C", "B", "A", "A", "C",
                        "C", "C", "B", "C", "A", "B", "B", "D", "A", "C", "B", "A", "C", "C", "A", "C", "A", "C", "B",
                        "B", "B", "B", "A", "C", "B", "C", "B", "C", "B", "B", "C", "D", "D", "A", "A", "B", "C", "C",
                        "C", "B", "C", "D", "A", "D", "C", "D", "B", "D", "D", "C", "B", "B", "D", "B", "B", "B", "C",
                        "A", "C", "B", "D", "C", "B", "C", "B", "D", "B", "C", "C", "D", "B", "B", "B", "A", "B", "B",
                        "B", "C", "A", "B", "C", "D", "B", "A", "B", "A", "B", "C", "C", "B", "B", "C", "B", "B", "A",
                        "A", "B", "D", "B", "C", "C", "B", "A", "B", "B", "B", "B", "B", "C", "B", "D", "B", "C", "A",
                        "B", "A", "C", "B", "D", "B", "B", "B", "C", "B", "B", "C", "B", "B", "B", "B", "D", "D", "B",
                        "B", "A", "C", "B", "D", "C", "A", "B", "C", "B", "D", "B", "D", "D", "B", "B", "B", "B", "B",
                        "A", "B", "A", "C", "A", "D", "A", "A", "D", "A", "B", "C", "C", "C", "D", "B", "B", "C", "D",
                        "C", "C", "C", "A", "B", "B", "D", "A", "B", "C", "C", "D", "B", "D", "C", "C", "B", "B", "C",
                        "C", "B", "A", "C", "D", "B", "A", "A", "B", "B", "A", "B", "C", "B", "B", "B", "D", "A", "B",
                        "D", "B", "C", "A", "B", "B", "B", "B", "B", "B", "A", "B", "A", "C", "A", "A", "B", "A", "B",
                        "B", "C", "C", "D", "B", "B", "C", "B", "D", "C", "C", "B", "D", "B", "B", "C", "D", "A", "B",
                        "D", "C", "C", "D", "D", "B", "B", "C", "D", "B", "B", "B", "A", "C", "D", "D", "C", "B", "A",
                        "B", "B", "D", "A", "C", "A", "C", "A", "B", "D", "D", "C", "B", "B", "A", "B", "A", "D", "B",
                        "C", "D", "B", "B", "A", "A", "A", "C", "A", "D", "B", "B", "A", "B", "D", "B", "B", "B", "A",
                        "D", "D", "C", "B", "D", "B", "B", "C", "A", "C", "A", "C", "B", "D", "B", "B", "B", "B", "C",
                        "C", "C", "C", "B", "A", "B", "B", "A", "C", "C", "C", "C", "A", "B", "A", "C", "C", "B", "D",
                        "C", "C", "D", "A", "D", "D", "A", "D", "B", "C", "A", "B", "A", "B", "C", "A", "C", "B", "B",
                        "C", "A", "B", "D", "B", "C", "B", "C", "B", "B", "C", "A", "B", "B", "A", "B", "B", "B", "C",
                        "D", "B", "A", "C", "B", "C", "A", "B", "B", "B", "B", "B", "D", "A", "D", "C", "B", "B", "D",
                        "B", "B", "B", "B", "B", "D", "A", "C", "B", "B", "B", "C", "B", "B", "B", "B", "C", "D", "B",
                        "A", "C", "D", "B", "A", "B", "B", "C", "B", "C", "A", "B", "C", "B", "C", "C", "B", "A", "B",
                        "D", "B", "B", "A", "A", "C", "C", "C", "B", "D", "C", "B", "C", "A", "C", "C", "C", "A", "C",
                        "B", "A", "B", "C", "C", "B", "C", "B", "B", "A", "C", "B", "C", "C", "D", "D", "B", "D", "C",
                        "A", "B", "C", "A", "B", "B", "B", "B", "A", "D", "C", "B", "C", "B", "B", "C", "C", "C", "C",
                        "D", "B", "C", "A", "C", "B", "A", "A", "C", "B", "B", "B", "D", "B", "B", "C", "B", "D", "B",
                        "A", "B", "C", "B", "D", "C", "D", "A", "C", "A", "C", "C", "D", "B", "B", "D", "C", "C", "B",
                        "D", "A", "D", "C", "D", "D", "B", "A", "C", "A", "D", "C", "C", "B", "C", "C", "B", "D", "B",
                        "B", "D", "B", "A", "B", "A", "D", "D", "B", "A", "C", "B", "C", "B", "C", "A", "A", "B", "B",
                        "A", "C", "B", "B", "A", "D", "D", "A", "A", "C", "B", "C", "D", "C", "B", "D", "A", "B", "A",
                        "B", "C", "D", "C", "C", "A", "C", "B", "C", "C", "B", "D", "A", "A", "C", "B", "B", "A", "B",
                        "B", "B", "B", "D", "A", "D", "D", "B", "B", "C", "D", "B", "D", "B", "B", "B", "B", "B", "C",
                        "D", "A", "D", "B", "C", "A", "C", "B", "B", "B", "D", "B", "B", "D", "A", "C", "B", "A", "B",
                        "C", "C", "D", "A", "A", "C", "B", "D", "C", "B", "B", "A", "C", "A", "B", "D", "D", "C", "C",
                        "D", "A", "C", "D", "B", "B", "B", "D", "B", "B", "B", "A", "B", "C", "A", "B", "B", "C", "B",
                        "B", "B", "D", "A", "D", "A", "A", "B", "B", "B", "B", "C", "D", "C", "B", "C", "D", "D", "A",
                        "A", "D", "C", "D", "B", "C", "D", "D", "B", "C", "A", "B", "B", "B", "B", "C", "D", "A", "B",
                        "A", "D", "B", "B", "D", "C", "C", "B", "C", "B", "C", "C", "B", "A", "A", "B", "C", "D", "C",
                        "C", "B", "C", "B", "C", "B", "C", "D", "D", "C", "C", "B", "B", "B", "D", "B", "B", "B", "C",
                        "A", "A", "D", "D", "D", "A", "D", "D", "B", "B", "A", "B", "D", "B", "D", "B", "A", "D", "C",
                        "B", "C", "B", "B", "B", "B", "C", "B", "D", "B", "C", "C", "A", "C", "C", "A", "B", "D", "C",
                        "C", "A", "D", "B", "B", "B", "D", "B", "D", "A", "B", "D", "D", "C", "D", "D", "D", "C", "B",
                        "C", "C", "C", "A", "D", "B", "D", "D", "D", "A", "D", "C", "C", "B", "B", "B", "B", "C", "C",
                        "D", "C", "C", "B", "B", "B", "C", "A", "B", "C", "A", "B", "B", "A", "D", "B", "D", "B", "C",
                        "D", "D", "C", "D", "C", "D", "B", "B", "C", "B", "B", "B", "C", "D", "B", "B", "A", "D", "C",
                        "A", "D", "A", "C", "D", "A", "B", "C", "C", "B", "C", "D", "D", "B", "B", "C", "D", "D", "B",
                        "C", "C", "B", "B", "D", "B", "C", "C", "D", "C", "B", "D", "B", "D", "D", "B", "A", "B", "C",
                        "A", "B", "B", "C", "C", "C", "C", "C", "B", "A", "C", "B", "B", "A", "D", "B", "C", "B", "C",
                        "C", "D", "C", "B", "B", "C", "B", "D", "B", "B", "D", "B", "D", "B", "C", "B", "A", "C", "A",
                        "B", "D", "A", "B", "C", "D", "C", "B", "C", "A", "A", "D", "B"]


def test_baseline(max_questions: int = -1, log_result: bool = True):
    start_time = time.time()

    correct_counter = 0
    results = []
    accuracies = []
    mismatches_with_lmql = []
    num_questions = len(dataset['id']) if max_questions == -1 else max_questions

    for question_id, question in enumerate(tqdm(dataset['question'][:num_questions])):
        if max_questions != -1 and question_id >= max_questions:
            break

        choices = dataset['choices'][question_id]
        labels = choices['label']  # [A, B, C, D]
        answers = choices['text']
        correct_answer = dataset['answerKey'][question_id]

        # keys_string = "[" + ", ".join(f"\"{key}\"" for key in keys) + "]"
        # prompt_tokens = model.tokenize(bytes(non_cot_decision_prompt(question, labels, answers)))

        prompt = non_cot_decision_prompt(question, labels, answers)

        response: CreateCompletionResponse = model.create_completion(
            prompt=prompt,
            temperature=0,
            logprobs=10,
            grammar=get_llama_grammar_from_labels(labels)
        )

        # {
        #   'id': 'cmpl-d6276de1-6c77-49bf-9eed-48d04212bb8b',
        #   'object': 'text_completion',
        #   'created': 1710360838,
        #   'model': '/Users/leonknauer/code/uni/thesis/models/TheBloke/Llama-2-7B-Chat-GGUF//llama-2-7b-chat.Q4_K_M.gguf',
        #   'choices': [{
        #       'text': 'C',
        #       'index': 0,
        #       'logprobs': {
        #           'tokens': ['C'],
        #           'text_offset': [362],
        #           'token_logprobs': [-14.748195],
        #           'top_logprobs': [{
        #               '\n': -0.03699828,
        #               ' (': -3.3310485,
        #               ' C': -8.365257,
        #               '©': -9.138457,
        #               '': -11.540381,
        #               ' The': -11.357604,
        #               ' Answer': -12.194313,
        #               '2': -12.286506,
        #               'C': -14.748195
        #         }]
        #      },
        #      'finish_reason': 'stop'
        #   }],
        #   'usage': {'prompt_tokens': 92, 'completion_tokens': 1, 'total_tokens': 93}}

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

        # if model_choice.strip() != lmql_answers[question_id]:  # Remove leading and trailing whitespace from model_choice
        #     mismatches_with_lmql.append({
        #         "question_id": question_id,
        #         "question": question,
        #         "answers": answers,
        #         "labels": labels,
        #         "model_choice": model_choice,
        #         "lmql_answers": lmql_answers[question_id],
        #         "top_logprobs": response['choices'][0]['logprobs']['top_logprobs'][0]
        #     })

        accuracies.append(round(correct_counter * 100/(question_id+1), 2))

        # print(f"-> {counter} of {question_id + 1} correct -> Accuracy {round(counter * 100/(question_id+1), 2)}%")
        # print("=================================================================")

    end_time = time.time()

    test_result = TestResult(
        model=model_filename,
        start_time=start_time,
        end_time=end_time,
        execution_seconds=round(time.time() - start_time, 2),
        total_accuracy=round(correct_counter * 100/(question_id+1), 2),
        accuracies=accuracies,
        results=results
    )

    # print(json.dumps(test_result, cls=TestResultEncoder, indent=4))
    # print(mismatches_with_lmql)

    if log_result:
        f = open(f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_baseline_arc_test_{model_filename}.json", "a")
        f.write(json.dumps(test_result, cls=NumpyEncoder, indent=4))
        f.close()

        f = open(f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_baseline_arc_test_mismatches_{model_filename}.json", "a")
        f.write(json.dumps(mismatches_with_lmql, cls=NumpyEncoder, indent=4))
        f.close()

    print(f"Execution took {round(time.time() - start_time, 2)} seconds.")


def is_equal(answer: str, reference: str):
    # TODO: define different rules for equlity

    return answer.strip() == reference


def get_llama_grammar_from_labels(labels: [str]) -> LlamaGrammar:
    labels_with_quotes = list(map(lambda label: f'"{label}"', labels))

    return LlamaGrammar.from_string(
        grammar=f"root   ::= [ ]? option \n"  # Accept single leading space
                f"option ::= ({'|'.join(labels_with_quotes) })",
        verbose=False
    )


def non_cot_decision_prompt(question: str, labels: [str], answers: [str]) -> str:
    options = ""

    for label_id, label in enumerate(labels):
        options += f"({label}) {answers[label_id]}\n"

    return f"Question: {question} \n" \
           f"Answer Choices: {options}" \
           f"Among {labels[0]} through {labels[-1]}, the answer is: "


if __name__ == '__main__':
    selected_model = "llama-2-7b-chat"

    model_filename = f"{selected_model}.Q4_K_M.gguf"
    model_path = f"{model_folder_path}/{model_filename}"

    if not os.path.isdir(model_folder_path):
        os.mkdir(model_folder_path)
        print("Model folder created, as it does not yet exist.")

    if not os.path.isfile(model_path):
        hf_hub_download(
            repo_id=models[selected_model]["hf-path"],
            filename=model_filename,
            local_dir=model_folder_path,
            local_dir_use_symlinks=True
        )

    model = Llama(
        model_path=model_path,
        n_gpu_layers=1000,
        n_ctx=8192,
        n_batch=1024,
        logits_all=True,
        verbose=False
    )
    print("Model loaded")

    dataset = load_dataset(
        path="ai2_arc",
        name="ARC-Challenge"  # ,
        # split="test",
        # num_proc=8,
        # keep_in_memory=True
    )
    dataset = dataset["test"]
    print("Dataset loaded") # TODO: Improve dataset load time (why faster in first draft?)

    test_baseline(max_questions=1, log_result=True)
