import json


def check_anomalies(file):
    file = open(file)
    data = json.load(file)
    results = data["results"]

    actually_correct_count = 0
    false_passes = 0
    false_fails = 0
    claimed_correct_count = 0
    error_count = 0

    for result in results:
        top_logprobs = result["top_logprobs"]
        model_choice = result["model_choice"]
        correct_answer = result["correct_answer"]
        labels = result["labels"]

        model_should_answer = None

        top_logprobs_sorted = sorted(top_logprobs.items(), key=lambda x: x[1], reverse=True)
        top_logprobs_sorted = dict(top_logprobs_sorted)
        filtered_top_logprobs = {k: v for k, v in top_logprobs_sorted.items() if k.strip() in labels}

        for logprob in top_logprobs_sorted:
            stripped_prediction = logprob.strip()  # remove spaces
            if stripped_prediction in labels:
                model_should_answer = stripped_prediction
                break  # use first (most likely) match

        if model_should_answer is None:
            print(f"Error! No fitting answer found in logprobs for question {result['question_id']}.")
            print(json.dumps(top_logprobs_sorted, indent=2))
            print("=" * 70)
            error_count += 1
        else:
            if model_should_answer == correct_answer:  # Model produced right answer with highest prob
                if not result["is_correct"]:  # But it selected something else - why?!
                    print(
                        f"Question ID {result['question_id']}: Model should have selected \"{model_should_answer}\" from below, but selected \"{model_choice}\", which lead to a FAIL.")
                    print(json.dumps(filtered_top_logprobs, indent=2))
                    print("=" * 70)
                    false_fails += 1
                else:  # It also selected this one
                    actually_correct_count += 1

            if result["is_correct"]:
                claimed_correct_count += 1
                if model_should_answer != model_choice.strip():  # Model claimed to be correct, but should have selected another non-correct label
                    print(
                        f"Question ID {result['question_id']}: Model should have selected \"{model_should_answer}\" from below, but selected \"{model_choice}\", which lead to a PASS.")
                    print(json.dumps(filtered_top_logprobs, indent=2))
                    print("=" * 70)
                    false_passes += 1

    print(f"{false_passes=}, {false_fails=}, {actually_correct_count=}, {claimed_correct_count=}, {error_count=}")


def compare(file1, file2):
    file1 = open(file1)
    file2 = open(file2)

    data1 = json.load(file1)
    data2 = json.load(file2)

    results1 = data1["results"]
    results2 = data2["results"]

    for result1 in results1:
        top_logprobs1 = result1["top_logprobs"]
        id1 = result1["question_id"]
        model_choice1 = result1["model_choice"]
        correct_answer1 = result1["correct_answer"]
        labels1 = result1["labels"]
        is_correct1 = result1["is_correct"]

        if results2[id1]["is_correct"] != is_correct1 or results2[id1]["model_choice"] != model_choice1:
            print("Mismatch in question {id}")


if __name__ == '__main__':
    check_anomalies(
        "../../results/2024-04-18_16-26-18_NonCoTStandardBenchmark_orca-2-13b_ARCChallengeDataset-1172_labels-unchanged_use-chat-template_LlamaCppPythonInferenceBackend_MBP-M1-Max.json")

# compare("../results/2024-04-08_15-19-28_non-cot-standard_arc-c-test-100_orca-2-13b.Q4_K_M.gguf_labels-unchanged_in-process_localhost.json", "../results/2024-04-08_15-22-18_non-cot-standard_arc-c-test-100_orca-2-13b.Q4_K_M.gguf_labels-unchanged_in-process_localhost.json")
