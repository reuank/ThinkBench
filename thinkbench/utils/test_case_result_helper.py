import json
import os
from pathlib import Path
from typing import List, Dict, Union

from benchmark.results import TestCaseResult, SingleBenchmarkResult, Metrics


class TestCaseResultHelper:
    @staticmethod
    def load_result(result_file_path: Union[Path, str]) -> TestCaseResult:
        if type(result_file_path) == str:
            result_file_path = Path(result_file_path)

        if result_file_path.is_dir():
            raise ValueError("Result file path is a directory.")
        elif result_file_path is None:
            raise ValueError("The result file path is invalid.")

        file = open(result_file_path)
        data = json.load(file)

        return TestCaseResultHelper.convert_to_test_case_result(data)

    @staticmethod
    def load_results(result_files_path: Union[Path, str]) -> List[TestCaseResult]:
        if type(result_files_path) == str:
            result_files_path = Path(result_files_path)

        result_files = []

        if result_files_path.is_dir():
            for file in sorted(os.listdir(result_files_path)):
                if file.endswith(".json"):
                    result_files.append(os.path.join(result_files_path, file))
        else:
            result_files.append(str(result_files_path))

        test_case_results: List[TestCaseResult] = []

        for result_file in result_files:
            result_file_path = Path(result_file)
            test_case_results.append(TestCaseResultHelper.load_result(result_file_path))

        return test_case_results

    @staticmethod
    def load_two_runs(
            first_result_files_path: Union[Path, str],
            second_result_files_path: Union[Path, str]
    ) -> (List[Dict], List[Dict]):
        first_test_case_results = TestCaseResultHelper.load_results(first_result_files_path)
        second_test_case_results = TestCaseResultHelper.load_results(second_result_files_path)

        if len(first_test_case_results) != len(second_test_case_results):
            raise ValueError("Number of files in the two dirs are not equal.")

        for result_id, first_test_case_result in enumerate(first_test_case_results):
            if len(first_test_case_result['results']) != len(second_test_case_results[result_id]['results']):
                raise ValueError("Both runs have a different number of results.")

        first_models = [result["model"] for result in first_test_case_results]
        second_models = [result["model"] for result in second_test_case_results]

        if sorted(first_models) != sorted(second_models):
            raise ValueError("No matching result present for each model.")

        return first_test_case_results, second_test_case_results

    @staticmethod
    def convert_to_test_case_result(data: Dict) -> TestCaseResult:
        if "label_permutation" not in data.keys():
            data["label_permutation"] = ""
        if "n_random" not in data.keys():
            data["n_random"] = -1
        if "random_seed" not in data.keys():
            data["random_seed"] = -1

        return TestCaseResult(
            uuid=data['uuid'],
            model=data['model'],
            dataset_name=data['dataset_name'],
            benchmark_name=data['benchmark_name'],
            label_numbering=data['label_numbering'],
            label_permutation=data['label_permutation'],
            n_random=data['n_random'],
            random_seed=data['random_seed'],
            hostname=data['hostname'],
            inference_backend=data['inference_backend'],
            inference_backend_properties=data.get('inference_backend_properties'),
            metrics=Metrics(**data['metrics']),
            start_time=data['start_time'],
            end_time=data['end_time'],
            execution_seconds=data['execution_seconds'],
            current_commit_hash=data['current_commit_hash'],
            comment=data['comment'],
            use_chat_template=data['use_chat_template'],
            results=[SingleBenchmarkResult(**result) for result in data['results']]
        )

    @staticmethod
    def all_results_contain_reasoning(test_case_results: List[TestCaseResult]) -> bool:
        return all("reasoning" in test_case_result["results"][0]["completions"][0] for test_case_result in test_case_results)

    @staticmethod
    def ensure_reasoning_present(test_case_results: List[TestCaseResult]):
        if not TestCaseResultHelper.all_results_contain_reasoning(test_case_results):
            raise ValueError("The CoT path contains result files that do not contain reasoning traces.")

    @staticmethod
    def get_model_choices(test_case_result: TestCaseResult) -> List[str]:
        return [single_benchmark_result["model_choice"] for single_benchmark_result in test_case_result["results"]]

    @staticmethod
    def get_completion_probs(test_case_result: TestCaseResult) -> List[Dict[str, float]]:
        return [single_result['completions'][0]['label']['logprobs'] for single_result in test_case_result["results"]]

    @staticmethod
    def get_model_choice_probs(test_case_result: TestCaseResult):
        model_choices = TestCaseResultHelper.get_model_choices(test_case_result)
        model_completion_probs = TestCaseResultHelper.get_completion_probs(test_case_result)

        assert len(model_choices) == len(model_completion_probs)

        model_choice_probs = []

        for index in range(len(model_choices)):
            # Try some variations of the label token
            model_choice_prob = model_completion_probs[index].get(model_choices[index], 0.0)
            if model_choice_prob == 0.0:
                model_choice_prob = model_completion_probs[index].get(" " + model_choices[index], 0.0)
            if model_choice_prob == 0.0:
                model_choice_prob = model_completion_probs[index].get(model_choices[index] + " ", 0.0)
            # if model_choice_prob == 0.0:
            #     Logger.error(f"Could not extract label prob for question id {index}, "
            #                  f"choice {model_choices[index]} and model {cot_test_case_result['model']}.")

            model_choice_probs.append(model_choice_prob)

        return model_choice_probs

    @staticmethod
    def get_label_probs(test_case_result: TestCaseResult) -> List[Dict[str, float]]:
        model_completion_probs = TestCaseResultHelper.get_completion_probs(test_case_result)

        label_probs: Dict[int, Dict[str, float]] = {}

        for single_benchmark_result_id, single_benchmark_result in enumerate(test_case_result["results"]):
            label_probs[single_benchmark_result_id] = {}
            label_completion = model_completion_probs[single_benchmark_result_id]

            valid_label_tokens = single_benchmark_result["labels"]
            valid_label_tokens.extend([" "+valid_label_token for valid_label_token in valid_label_tokens])

            for valid_label_token in valid_label_tokens:
                if valid_label_token in label_completion.keys():
                    label_probs[single_benchmark_result_id][valid_label_token] = label_completion[valid_label_token]

        return [probs for question_id, probs in label_probs.items()]

    @staticmethod
    def get_label_completion_top_tokens(test_case_result: TestCaseResult) -> Dict[str, List[int]]:
        top_tokens: Dict[str, List[int]] = {}

        completion_probs_list = TestCaseResultHelper.get_completion_probs(test_case_result)

        for completion_id, completion_probs in enumerate(completion_probs_list):
            for token, prob in completion_probs.items():
                if token in top_tokens.keys():
                    top_tokens[token].append(completion_id)
                else:
                    top_tokens[token] = [completion_id]

        return top_tokens

    @staticmethod
    def get_correct_answers(test_case_result: TestCaseResult) -> List[str]:
        return [single_benchmark_result["correct_answer"] for single_benchmark_result in test_case_result["results"]]

    @staticmethod
    def get_question_ids_of_correct_model_choices(
            test_case_result: TestCaseResult
    ):
        question_ids_of_correct_model_choices = []
        for single_benchmark_result in test_case_result["results"]:
            if single_benchmark_result["is_correct"]:
                question_ids_of_correct_model_choices.append(single_benchmark_result["question_id"])

        return question_ids_of_correct_model_choices

    @staticmethod
    def get_question_ids_of_incorrect_model_choices(
            test_case_result: TestCaseResult
    ):
        question_ids_of_incorrect_model_choices = []

        for single_benchmark_result in test_case_result["results"]:
            if not single_benchmark_result["is_correct"]:
                question_ids_of_incorrect_model_choices.append(single_benchmark_result["question_id"])

        return question_ids_of_incorrect_model_choices
