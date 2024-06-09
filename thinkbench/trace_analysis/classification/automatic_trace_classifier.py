import json
import re
import string
from pathlib import Path
from string import Template
from typing import List, Dict

import nltk
from tqdm import tqdm

from benchmark.results import TestCaseResult, SingleBenchmarkResult
from storage.backends.json_file_storage import JsonFileStorage
from trace_analysis.classification.classification_result import ClassificationResult, SingleClassification
from trace_analysis.classification.trace_class import TraceClass
from trace_analysis.classification.trace_classifier import TraceClassifier
from utils.logger import Logger


class AutomaticTraceClassifier(TraceClassifier):
    @staticmethod
    def classify_test_case_results(
            cot_test_case_results: List[TestCaseResult],
            non_cot_test_case_results: List[TestCaseResult],
            override: bool
    ) -> List[ClassificationResult]:
        json_storage_backend: JsonFileStorage = JsonFileStorage()
        classification_results: List[ClassificationResult] = []

        # Loop over Test Case Results
        for cot_test_case_result_id, cot_test_case_result in enumerate(
                tqdm(cot_test_case_results, desc="Running automatic trace classification.")
        ):
            non_cot_test_case_result = non_cot_test_case_results[cot_test_case_result_id]

            classification_result_file_name = JsonFileStorage.get_classification_result_file_name_for_test_cases(
                cot_test_case_result=cot_test_case_result,
                non_cot_test_case_result=non_cot_test_case_result
            )

            current_classification_result = None

            classification_result_file_path: Path = json_storage_backend.analysis_path / classification_result_file_name
            if classification_result_file_path.is_file():
                current_classification_result = json_storage_backend.load_classification_result(
                    classification_result_file_name=classification_result_file_name
                )

            new_automatic_classification = AutomaticTraceClassifier.classify_test_case_result(
                cot_test_case_result=cot_test_case_result,
                non_cot_test_case_result=non_cot_test_case_result
            )

            classification_results.append(
                TraceClassifier.merge_manual_class_ids(
                    manual_classification_result=current_classification_result,
                    automatic_classification_result=new_automatic_classification
                )
            )

        return classification_results

    @staticmethod
    def classify_test_case_result(
            cot_test_case_result: TestCaseResult,
            non_cot_test_case_result: TestCaseResult
    ) -> ClassificationResult:
        single_classifications: List[SingleClassification] = []

        for cot_single_benchmark_result_id, cot_single_benchmark_result in enumerate(cot_test_case_result["results"]):
            non_cot_single_benchmark_result = non_cot_test_case_result["results"][cot_single_benchmark_result_id]

            single_classification = AutomaticTraceClassifier.classify_single_benchmark_result(
                model_name=cot_test_case_result["model"],
                cot_single_benchmark_result=cot_single_benchmark_result,
                non_cot_single_benchmark_result=non_cot_single_benchmark_result
            )

            single_classifications.append(single_classification)

        return ClassificationResult(
            non_cot_uuid=non_cot_test_case_result["uuid"],
            cot_uuid=cot_test_case_result["uuid"],
            model=cot_test_case_result["model"],
            dataset_name=cot_test_case_result["dataset_name"],
            cot_benchmark_name=cot_test_case_result["benchmark_name"],
            non_cot_benchmark_name=non_cot_test_case_result["benchmark_name"],
            results=single_classifications
        )

    @staticmethod
    def classify_single_benchmark_result(
            model_name: str,
            cot_single_benchmark_result: SingleBenchmarkResult,
            non_cot_single_benchmark_result: SingleBenchmarkResult,
    ) -> SingleClassification:
        return AutomaticTraceClassifier.classify_single_reasoning_trace(
            model_name=model_name,
            cot_model_choice=cot_single_benchmark_result['model_choice'],
            question_id=cot_single_benchmark_result['question_id'],
            valid_labels=cot_single_benchmark_result['labels'],
            non_cot_model_choice=non_cot_single_benchmark_result['model_choice'],
            reasoning_trace=cot_single_benchmark_result['completions'][0]['reasoning']['text']
        )

    @staticmethod
    def classify_single_reasoning_trace(
            model_name: str,
            valid_labels: List[str],
            question_id: int,
            cot_model_choice: str,
            non_cot_model_choice: str,
            reasoning_trace: str
    ) -> SingleClassification:
        trace_sentences = AutomaticTraceClassifier.get_trace_sentences(reasoning_trace)
        trace_answer_sentences, sentence_counts = AutomaticTraceClassifier.extract_answer_sentences(
            model_name=model_name,
            trace_sentences=trace_sentences,
            labels=valid_labels,
            debug=False  # (model_name == "orca-2-13b" and question_id == 147)
        )

        single_classification: SingleClassification = SingleClassification(
            question_id=question_id,
            reasoning=reasoning_trace,
            cot_model_choice=cot_model_choice,
            non_cot_model_choice=non_cot_model_choice,
            manual_class_id=None,
            automatic_class_id=4,
            extracted_labels=[]
        )

        # At least one answer sentence was found
        if len(trace_answer_sentences) > 0:
            extracted_labels = []
            for trace_answer_sentence in trace_answer_sentences:
                extracted_labels += AutomaticTraceClassifier.extract_label_matches_from_sentence(
                    trace_answer_sentence,
                    valid_labels
                )
            extracted_labels = sorted(set(extracted_labels))

            if len(extracted_labels) > 1:
                single_classification["automatic_class_id"] = TraceClass.TRACE_LABEL_AMBIGUOUS.value

            elif len(extracted_labels) == 0 and sentence_counts["none_picked"] > 0:
                single_classification["automatic_class_id"] = TraceClass.NO_TRACE_LABEL.value

            elif len(extracted_labels) == 1:
                if cot_model_choice == extracted_labels[0]:
                    single_classification[
                        "automatic_class_id"] = TraceClass.TRACE_LABEL_UNAMBIGUOUS_EXTRACTION_SUCCEEDED.value
                else:
                    single_classification[
                        "automatic_class_id"] = TraceClass.TRACE_LABEL_UNAMBIGUOUS_EXTRACTION_FAILED.value

            single_classification["extracted_labels"] = extracted_labels

        elif sentence_counts["questions_commands"] > 0:
            single_classification["automatic_class_id"] = TraceClass.NO_TRACE_LABEL.value

        else:
            if sentence_counts["none_picked"] > 0:
                single_classification["automatic_class_id"] = TraceClass.NO_TRACE_LABEL.value

        return single_classification

    @staticmethod
    def extract_answer_sentences(
        model_name: str,
        trace_sentences: List[str],
        labels: List[str],
        debug: bool = False
    ) -> (List[str], Dict[str, int]):
        trace_answer_sentences = []
        sentence_counts = {
            "questions_commands": 0,
            "none_picked": 0,
        }

        in_sentence_chars = string.ascii_letters + string.digits + string.whitespace
        printable_chars = string.ascii_letters + string.digits + string.punctuation + string.whitespace

        definite_answer_sentence_indicators = []

        answer_sentence_indicators = [
            r"the correct answer is",
            r"that's the correct answer",
            fr"the correct [{re.escape(in_sentence_chars)}]+ is",
            r"the correct answer among",
            r"The correct answer is",
            r"is the correct answer",
            r"the best answer",
            r"The best answer",
            r"the best description",
            r"the best choice",
            r"the best example",
            r"The best conclusion",
            r"the most plausible conclusion",
            r"the most likely",
            r"the answer is",
            r"The answer is",
            r"Answer:",
            r"Correct answer:",
            # r"is correct",
            r"Option \([A-Z0-9]\) is correct",
            r"option \([A-Z0-9]\) is correct",
            r"Option [A-Z0-9] is correct",
            r"option [A-Z0-9] is correct",
            r"is the most likely explanation",
            r"is the most likely answer",
            r"is the most logical explanation",
            r"This option is the best",

            # Orca
            r"Based on the",
            fr"Based on [{re.escape(in_sentence_chars)}]+ analysis",
            fr"Based on [{re.escape(in_sentence_chars)}]+ reasoning",
            fr"Based on [{re.escape(in_sentence_chars)}]+ information",
            r"Comparing the answer choices",
            r"Comparing the remaining choices",
            r"### Final Answer:",
            r"#Answer",

            # Mistral
            r"Therefore, the",
            r"Given the answer choices, the"
        ]

        after_answer_sentence_indicators = [
            "This option is correct",
            "This is correct",
            "This is the correct answer",
            "This answer choice is the correct one",
            "Therefore, it is the correct answer.",
            "Correct!"
        ]

        exclusion_indicators = [
            r"is incorrect",
            r"not the best choice",
            r"not the correct",
            r"may not be the best choice",
            r"not the best",

            # Orca
            r"not the most likely",
            r"we can eliminate",
            r"answer is not",

            # Llama 2 70b
            r"the correct answer cannot be"
        ]

        none_picked_indicators = [
            r"None of the given",
            r"no single correct answer",
            rf"none of the [{re.escape(in_sentence_chars)}]* answer choices",
            r"none of the answer choices",
            rf"none of the options are correct",
            r"error in the given answer choices",
            r"no correct answer among the given choices",
            r"we cannot determine",
            r"Cannot be determined"
        ]

        question_command_indicators = [
            rf"What is [{re.escape(in_sentence_chars)}]+ answer?",
            rf"What do you think?",
            rf"Please [{re.escape(in_sentence_chars)}]+.",
            rf"Which answer [{re.escape(in_sentence_chars)}]+?",
            rf"Can you [{re.escape(in_sentence_chars)}]+?"
        ]

        if "orca" in model_name:
            # tokenizer._params.abbrev_types.remove({"a", "b", "c", "d"})
            definite_answer_sentence_indicators.append("### Final Answer")
        elif "mistral" in model_name:
            exclusion_indicators.append(r"rather than")

        if debug:
            enumerated_sentences = [
                f"ID {trace_sentence_id}) {trace_sentence}"
                for trace_sentence_id, trace_sentence in enumerate(trace_sentences)
            ]
            Logger.debug(f"Analyzing trace sentences:\n{json.dumps(enumerated_sentences, indent=2)}")

        # Loop over all sentences of the reasoning trace
        for trace_sentence_id, trace_sentence in enumerate(trace_sentences):
            # Skip sentences which are most likely not relevant
            if any(re.search(exclusion_indicator, trace_sentence, re.IGNORECASE) for exclusion_indicator in exclusion_indicators):
                if debug:
                    Logger.debug(f"--> Sentence {trace_sentence_id} skipped because of exclusion_indicator")
                continue

            # Check for matches with after_answer_sentence_indicator
            if any(after_answer_sentence_indicator in trace_sentence for after_answer_sentence_indicator in after_answer_sentence_indicators):  # Include sentence before
                # Indicator stands alone, not e.g. "Option (B) improve existing products: This option is correct!"
                if len(AutomaticTraceClassifier.extract_label_matches_from_sentence(trace_sentence, labels)) == 0:
                    if debug:
                        Logger.debug(f"--> After answer indicator found in sentence id {trace_sentence_id}")
                    trace_answer_sentences.append(trace_sentences[trace_sentence_id - 1] + "\n" + trace_sentence)
                    continue

            # If sentence is not skipped until here, search for none picked indicators
            if any(re.search(none_picked_indicator, trace_sentence, re.IGNORECASE) for none_picked_indicator in none_picked_indicators):
                if debug:
                    Logger.debug(f"--> None picked indicator found in sentence id {trace_sentence_id}, skipping.")
                # trace_answer_sentences.append(trace_sentence)
                sentence_counts["none_picked"] += 1
                continue

            # If sentence is not skipped until here, search for answer indicators
            if any(re.search(answer_sentence_indicator, trace_sentence, re.IGNORECASE) for answer_sentence_indicator in answer_sentence_indicators):
                if debug:
                    Logger.debug(f"--> Answer sentence indicator found in sentence id {trace_sentence_id}.")
                trace_answer_sentences.append(trace_sentence)

            # The last sentence contains a question or a command, so the model probably did not choose a label
            if trace_sentence_id == len(trace_sentences) - 1 and any(re.search(question_command_indicator, trace_sentence) for question_command_indicator in question_command_indicators):
                if debug:
                    Logger.debug(f"--> Question or command indicator found in sentence id {trace_sentence_id}.")
                # trace_answer_sentences.append(trace_sentence)
                sentence_counts["questions_commands"] += 1

        for trace_answer_sentence in trace_answer_sentences:
            if any(re.search(definite_answer_sentence_indicator, trace_answer_sentence, re.IGNORECASE) for definite_answer_sentence_indicator in definite_answer_sentence_indicators):
                if debug:
                    Logger.debug(f"--> Definite answer sentence indicator found in answer sentence '{trace_answer_sentence}'.")
                trace_answer_sentences = [trace_answer_sentence]
                break

        if debug:
            Logger.debug(f"All extracted trace answer sentences:\n{json.dumps(trace_answer_sentences, indent=2)}")

        return trace_answer_sentences, sentence_counts

    @staticmethod
    def get_trace_sentences(reasoning_trace: str, split_paragraphs: bool = True):
        # nltk.download("punkt")
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

        trace_sentences = []

        if split_paragraphs:
            # Step 1: Split the text by double linebreaks
            paragraphs = reasoning_trace.split("\n\n")

            # Step 2: Merge paragraphs that belong together
            merged_paragraphs = []
            i = 0
            while i < len(paragraphs):
                if paragraphs[i].endswith(":") and i + 1 < len(paragraphs):
                    # Merge current sentence with the next one
                    merged_paragraphs.append(paragraphs[i] + " " + paragraphs[i + 1])
                    i += 2  # Skip the next sentence as it is already merged
                else:
                    # Append the current sentence as is
                    merged_paragraphs.append(paragraphs[i])
                    i += 1

            # Step 3: Tokenize sentences within each paragraph
            for paragraph in merged_paragraphs:
                sentences = tokenizer.tokenize(paragraph)
                trace_sentences.extend(sentences)
        else:
            trace_sentences = tokenizer.tokenize(reasoning_trace)

        return trace_sentences

    @staticmethod
    def extract_label_matches_from_sentence(sentence: str, labels: List[str]) -> List[str]:
        label_patterns = [
            Template("(${label})"),
            Template(" ${label})"),
            Template(" ${label} "),
            Template("${label}. "),
            Template(" ${label}."),
            Template("option ${label}"),
            Template("Option ${label}"),

            # Mistral 7b instr.
            Template(": ${label},"),  # So, the answer is: D,
            Template("${label}:"),  # So, the most likely answer is C:

            # Orca 2 13b
            Template("Final Answer: ${label}"),
        ]

        label_exclusion_patterns = [
            Template("not (${label})"),  # the answer is not (D)
        ]

        matches = {}

        # Fix distracting wordings
        sentence = sentence.replace(f"among {labels[0]} through {labels[-1]}", "")
        sentence = sentence.replace(f"Among {labels[0]} through {labels[-1]}", "")
        for label in labels:
            sentence = sentence.replace(f"({label}) A ",
                                        f"({label})")  # fixes e.g. "(D) A buildup of cooled lava: This is the correct answer!"
            sentence = sentence.replace(f"Option {label}: A ",
                                        f"Option ({label})")  # fixes e.g. "So, the best example of physical weathering is Option B: A bulldozer pushing soil."
            sentence = sentence.replace(f"{label}) A ",
                                        f"{label})")  # fixes e.g. Therefore, the correct answer is C) A large number of...

        for label in labels:
            for label_pattern in label_patterns:
                if f"the correct answer is ({label})" in sentence and "or" not in sentence:  # fixes "So, the correct answer is (A) or (B)"
                    return [label]
                if label_pattern.substitute(label=label) in sentence:
                    if label not in matches.keys():
                        matches.update({label: []})
                    matches[label].append(label_pattern.substitute(label=label))

        return list(matches.keys())
