import re
import string
from enum import Enum
from string import Template
from typing import List, Dict

import nltk
from tqdm import tqdm

from trace_analysis.trace_classifier import TraceClassifier


class TraceClass(Enum):
    TRACE_LABEL_UNAMBIGUOUS_EXTRACTION_SUCCEEDED = 1
    TRACE_LABEL_UNAMBIGUOUS_EXTRACTION_FAILED = 2
    TRACE_LABEL_AMBIGUOUS = 3
    NO_TRACE_LABEL = 4

    @staticmethod
    def get_ids():
        return [trace_class.value for trace_class in TraceClass]


class AutomaticTraceClassifier(TraceClassifier):
    @staticmethod
    def is_trace_class(value) -> bool:
        try:
            value = int(value)
            for trace_class in TraceClass:
                if trace_class.value == value:
                    return True
            return False
        except ValueError:
            return False

    @staticmethod
    def get_model_choices(data: Dict) -> List[str]:
        return [single_result["model_choice"] for single_result in data["results"]]

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

        for label in labels:
            for label_pattern in label_patterns:
                if f"the correct answer is ({label})" in sentence and "or" not in sentence:  # fixes "So, the correct answer is (A) or (B)"
                    return [label]
                if label_pattern.substitute(label=label) in sentence:
                    if label not in matches.keys():
                        matches.update({label: []})
                    matches[label].append(label_pattern.substitute(label=label))

        # if len(matches.keys()) > 1:
        #     print("="*40)
        #     print("More than one label was suggested!")
        #     print(matches)
        #     print(sentence)
        # elif len(matches.keys()) < 1:
        #     print("=" * 40)
        #     print("No label could be extracted!")
        #     print(f"Sentence: {sentence}")

        return list(matches.keys())

    @staticmethod
    def classify_traces(all_cot_results: List[Dict], all_non_cot_results: List[Dict]) -> Dict[str, Dict]:
        classification_results: Dict[str, Dict] = {}

        for cot_result_id, cot_result in enumerate(tqdm(all_cot_results, desc="Running automatic trace classification.")):
            non_cot_result = all_non_cot_results[cot_result_id]
            non_cot_choices = AutomaticTraceClassifier.get_model_choices(non_cot_result)

            model_name = cot_result["model"]
            cot_uuid = cot_result["uuid"]
            non_cot_uuid = non_cot_result["uuid"]

            trace_ids = {
                "extractable": [],
                "ambiguous": [],
                "questions": [],
                "unresolved": [],
            }

            table_rows = []

            # nltk.download("punkt")
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

            definite_answer_sentence_indicators = []

            in_sentence_chars = string.ascii_letters + string.digits + string.whitespace
            printable_chars = string.ascii_letters + string.digits + string.punctuation + string.whitespace

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

                # Mistral
                r"rather than"
            ]

            none_picked_indicators = [
                # Orca 2 13b
                r"None of the given",
                r"no single correct answer",
                rf"none of the [{re.escape(in_sentence_chars)}]* answer choices",
                r"none of the answer choices",
                rf"none of the options are correct",
                r"error in the given answer choices",
                r"no correct answer among the given choices",
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

            for cot_result_row_id, cot_result_row in enumerate(cot_result["results"]):
                non_cot_result_row = non_cot_result["results"][cot_result_row_id]
                reasoning_trace = cot_result_row["completions"][0]["reasoning"]["text"]
                split_paragraphs = True

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
                    trace_sentences = []
                    for paragraph in merged_paragraphs:
                        sentences = tokenizer.tokenize(paragraph)
                        trace_sentences.extend(sentences)
                else:
                    trace_sentences = tokenizer.tokenize(reasoning_trace)

                trace_answer_sentences = []
                question_command_sentences = []
                none_picked_sentences = []

                # Loop over all sentences of the reasoning trace
                for trace_sentence_id, trace_sentence in enumerate(trace_sentences):
                    # Skip sentences which are most likely not relevant
                    if any(re.search(exclusion_indicator, trace_sentence, re.IGNORECASE) for exclusion_indicator in
                           exclusion_indicators):
                        continue

                    # Check for matches with after_answer_sentence_indicator
                    if any(after_answer_sentence_indicator in trace_sentence for after_answer_sentence_indicator in
                           after_answer_sentence_indicators):  # Include sentence before

                        # Indicator stands alone, not e.g. "Option (B) improve existing products: This option is correct!"
                        if len(AutomaticTraceClassifier.extract_label_matches_from_sentence(trace_sentence, cot_result_row["labels"])) == 0:
                            trace_answer_sentences.append(trace_sentences[trace_sentence_id - 1] + "\n" + trace_sentence)
                            continue

                    # If sentence is not skipped until here, search for answer indicators
                    if any(re.search(answer_sentence_indicator, trace_sentence, re.IGNORECASE) for answer_sentence_indicator
                           in answer_sentence_indicators):
                        trace_answer_sentences.append(trace_sentence)

                    # If sentence is not skipped until here, search for answer indicators
                    if any(re.search(none_picked_indicator, trace_sentence, re.IGNORECASE) for none_picked_indicator in
                           none_picked_indicators):
                        # trace_answer_sentences.append(trace_sentence)
                        none_picked_sentences.append(trace_sentence)

                    # The last sentence contains a question or a command, so the model probably did not chose a label
                    if trace_sentence_id == len(trace_sentences) - 1 and any(
                            re.search(question_command_indicator, trace_sentence) for question_command_indicator in
                            question_command_indicators):
                        # trace_answer_sentences.append(trace_sentence)
                        question_command_sentences.append(trace_sentence)

                for trace_answer_sentence in trace_answer_sentences:
                    if any(re.search(definite_answer_sentence_indicator, trace_answer_sentence, re.IGNORECASE) for
                           definite_answer_sentence_indicator in definite_answer_sentence_indicators):
                        trace_answer_sentences = [trace_answer_sentence]
                        break

                question_id = cot_result_row["question_id"]

                table_row = {
                    "question_id": cot_result_row["question_id"],
                    "reasoning": reasoning_trace,
                    "correct_answer": cot_result_row["correct_answer"],
                    "cot_choice": cot_result_row["model_choice"],
                    "non_cot_choice": non_cot_choices[cot_result_row_id],
                    "automatic_class_id": 4,
                    "extracted_labels": ""
                }

                # At least one answer sentence was found
                if len(trace_answer_sentences) > 0:
                    extracted_labels = []
                    for trace_answer_sentence in trace_answer_sentences:
                        extracted_labels += AutomaticTraceClassifier.extract_label_matches_from_sentence(
                            trace_answer_sentence,
                            cot_result_row["labels"]
                        )
                    extracted_labels = sorted(set(extracted_labels))

                    if len(extracted_labels) > 1:
                        table_row.update(automatic_class_id=TraceClass.TRACE_LABEL_AMBIGUOUS.value)
                        trace_ids["ambiguous"].append(question_id)

                    elif len(extracted_labels) == 0 and len(none_picked_sentences) > 0:
                        table_row.update(automatic_class_id=TraceClass.NO_TRACE_LABEL.value)

                    elif len(extracted_labels) == 1:
                        if cot_result_row["model_choice"] == extracted_labels[0]:
                            table_row.update(automatic_class_id=TraceClass.TRACE_LABEL_UNAMBIGUOUS_EXTRACTION_SUCCEEDED.value)
                        else:
                            table_row.update(automatic_class_id=TraceClass.TRACE_LABEL_UNAMBIGUOUS_EXTRACTION_FAILED.value)

                    table_row.update(extracted_labels=",".join(extracted_labels))
                    trace_ids["extractable"].append(question_id)

                elif len(question_command_sentences) > 0:
                    table_row.update(automatic_class_id=TraceClass.NO_TRACE_LABEL.value)
                    trace_ids["questions"].append(question_id)

                else:
                    if len(none_picked_sentences) > 0:
                        table_row.update(automatic_class_id=TraceClass.NO_TRACE_LABEL.value)
                    trace_ids["unresolved"].append(question_id)

                table_rows.append(table_row)

            classification_results.update({
                model_name: {
                    "cot_uuid": cot_uuid,
                    "non_cot_uuid": non_cot_uuid,
                    "result_rows": table_rows,
                }
            })

        return classification_results
