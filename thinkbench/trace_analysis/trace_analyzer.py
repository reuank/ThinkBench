import re
import json
import datetime
import string
from enum import Enum
from string import Template

import random
import nltk.data
import xlsxwriter
from typing import List, Dict

import fire

from utils import result_loader


class SingleResult:
    def __init__(self, data):
        self.data = data


class Error(Enum):
    AMBIGUOUS = "Ambiguous"
    NO_LABEL_PICKED = "No label picked"
    WRONG_LABEL_PICKED = "Wrong label picked"
    QUESTION = "Question / Command"
    EXTRACTION_FAILED = "Extraction failed"
    OTHER_ERROR = "Other error"


class TraceAnalyzer:
    @staticmethod
    def analyze(method: str, result_file: str, baseline_result_file: str = None):
        baseline_result_file_data = None
        if baseline_result_file:
            baseline_result_file_data = result_loader.load_result_file(baseline_result_file)

        result_file_data = result_loader.load_result_file(result_file)

        analysis_result = TraceAnalyzer.method_mapping[method](result_file_data, baseline_result_file_data)

        TraceAnalyzer.write_to_xlsx(analysis_result)
        TraceAnalyzer.write_to_json_file(data=analysis_result, suffix=f"{analysis_result['model_name']}_analysis")

        samples = TraceAnalyzer.pick_samples(result=analysis_result, n_samples=100, seed=analysis_result["model_name"])
        TraceAnalyzer.write_to_xlsx(samples, samples=True)

    @staticmethod
    def get_single_results(data: Dict) -> List[SingleResult]:
        return [SingleResult(single_result) for single_result in data["results"]]

    @staticmethod
    def get_model_choices(data: Dict) -> List[str]:
        return [single_result["model_choice"] for single_result in data["results"]]

    @staticmethod
    def pick_samples(result: Dict, n_samples: int = 100, seed: str = "") -> Dict:
        results = result["result_rows"]
        n_results = len(results)

        random.seed(seed)
        samples = [results[random.randint(0, n_results)] for i in range(0, n_samples)]

        assert len(samples) == n_samples

        result.update(result_rows=samples)

        return result

    @staticmethod
    def write_to_json_file(data, suffix: str = ""):
        filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{suffix}.json"
        f = open(filename, "a")
        f.write(json.dumps(data, indent=2))

        print(f"File {filename} written.")

    @staticmethod
    def write_to_xlsx(analysis_result: Dict, suffix: str = "", samples: bool = False):
        padding_top_left = 1

        cols = [
            "question_id",
            "question",
            "answer_options",
            "correct_answer",

            "reasoning",
            "answer_sentences",
            "automatic_extraction",
            "manual_extraction",
            "error",
            "trace_label_correct",

            "model_choice",
            "model_choice_correct",
            "labels_match",

            "baseline_model_choice",
            "baseline_model_choice_correct",
            "runs_match",

            "comment"
        ]

        if samples:
            cols = cols[0:9]
            cols.remove("manual_extraction")
            cols.append("correct_classification")

        def col_id(col_name: str, ignore_padding: bool = False):
            if col_name not in cols:
                return -1

            col_id = cols.index(col_name)
            if not ignore_padding:
                col_id += 1

            return col_id

        def col_letter(col_name: str):
            if col_id(col_name, ignore_padding=False) == -1:
                raise ValueError("col not found")

            return chr(65 + col_id(col_name, ignore_padding=False))

        pastel_red = "#FFCCCC"
        pastel_green = "#CCFFCC"

        model_name = analysis_result["model_name"]
        cot_uuid = analysis_result["cot_uuid"]
        baseline_uuid = analysis_result["baseline_uuid"]

        workbook = xlsxwriter.Workbook(f"{model_name}_{cot_uuid[:8] + ('_' + baseline_uuid[:8] if baseline_uuid != '' else '')}_{'samples' if samples else 'analysis'}.xlsx")
        worksheet = workbook.add_worksheet(model_name)
        worksheet.protect()

        cell_format = workbook.add_format({
            "align": "left",
            "valign": "top",
            "text_wrap": True,
            "num_format": "@"
        })

        cell_format_unlocked = workbook.add_format({
            "align": "left",
            "valign": "top",
            "text_wrap": True,
            "locked": False,
            "num_format": "@"
        })

        thick_border_right_format = workbook.add_format({
            "align": "left",
            'valign': 'top',
            'text_wrap': True,
            'right': 5,
            "num_format": "@"
        })

        cell_format_unlocked_thick_border_left = workbook.add_format({
            "align": "left",
            "valign": "top",
            "text_wrap": True,
            "locked": False,
            'left': 5,
            "num_format": "@"
        })

        # Set column widths
        for col_num, col_data in enumerate(cols):
            worksheet.set_column(col_num + padding_top_left, col_num + padding_top_left, 10)
        worksheet.set_column(f"{col_letter('question')}:{col_letter('question')}", 30)
        worksheet.set_column(f"{col_letter('answer_options')}:{col_letter('question')}", 30)
        worksheet.set_column(f"{col_letter('reasoning')}:{col_letter('reasoning')}", 60)
        worksheet.set_column(f"{col_letter('answer_sentences')}:{col_letter('answer_sentences')}", 30)
        if not samples:
            worksheet.set_column(f"{col_letter('comment')}:{col_letter('comment')}", 30)

        # Write table content
        for row_num, row_data in enumerate(analysis_result["result_rows"]):
            if samples:
                row_data.update(correct_classification="")

            for col_num, (col_key, cell_data) in enumerate(row_data.items()):
                if col_key in cols:
                    col_num = cols.index(col_key)

                    if col_num == col_id("correct_answer", True):
                        worksheet.write(row_num + 2, col_num + 1, cell_data, thick_border_right_format)
                    elif col_num == col_id("reasoning", True):
                        worksheet.write(row_num + 2, col_num + 1, cell_data, cell_format_unlocked)
                    elif col_num == col_id("trace_label_correct", True):
                        worksheet.write_formula(row_num + 2, col_num + 1, f'=IF(OR('
                                                                          f'${col_letter("automatic_extraction")}{row_num + 3}=${col_letter("correct_answer")}{row_num + 3}, '
                                                                          f'${col_letter("manual_extraction")}{row_num + 3}=${col_letter("correct_answer")}{row_num + 3}), 1, 0)', thick_border_right_format)
                    elif col_num == col_id("labels_match", True):
                        worksheet.write_formula(row_num + 2, col_num + 1, f'=IF(OR(${col_letter("automatic_extraction")}{row_num + 3}=${col_letter("model_choice")}{row_num + 3}, ${col_letter("manual_extraction")}{row_num + 3}=${col_letter("model_choice")}{row_num + 3}), 1, 0)', thick_border_right_format)
                    elif col_num == col_id("runs_match", True):
                        worksheet.write_formula(row_num + 2, col_num + 1, f'=IF(${col_letter("baseline_model_choice")}{row_num + 3}=${col_letter("model_choice")}{row_num + 3}, 1, 0)', thick_border_right_format)
                    elif (col_num == col_id("manual_extraction", True) or col_num == col_id("error", True) or col_num == col_id("comment", True)) and not samples:
                        worksheet.write(row_num + 2, col_num + 1, cell_data, cell_format_unlocked)
                    elif col_num == col_id("correct_classification", True):
                        worksheet.write(row_num + 2, col_num + 1, cell_data, cell_format_unlocked_thick_border_left)
                    else:
                        worksheet.write(row_num + 2, col_num + 1, cell_data, cell_format)

        # Define the table range (start_row, start_col, end_row, end_col)
        start_row = padding_top_left
        start_col = padding_top_left
        end_row = len(analysis_result["result_rows"]) + padding_top_left + 1  # 1 space, 1 header
        end_col = len(cols)

        # Add the table with headers
        worksheet.add_table(start_row, start_col, end_row, end_col, {
            "total_row": 1,
            'columns': [
                {'header': col, 'total_function': 'count'} for col in cols
            ]
        })

        for col in cols:
            if col in ["trace_label_correct", "model_choice_correct", "labels_match", "baseline_model_choice_correct", "runs_match"]:
                worksheet.write_formula(end_row + 1, col_id(col), f'=COUNTIF({col_letter(col)}2:{col_letter(col)}{end_row + 1}, "1")')

        worksheet.data_validation(
            start_row + 1, col_id("error"), end_row, col_id("error"),
            {
                'validate': 'list',
                'source': [error.value for error in Error]
            }
        )

        for row_num in range(start_row + 1, end_row):  # Skip header and total row
            # Color answer sentences dark red if they are empty
            worksheet.conditional_format(
                row_num, col_id("answer_sentences"),
                row_num, col_id("answer_sentences"),
                {
                    'type': 'formula',
                    'criteria': f'=${col_letter("answer_sentences")}{row_num + 1}=""',
                    'format': workbook.add_format({'bg_color': 'red'})
                }
            )

            # Color error yellow if set
            worksheet.conditional_format(
                row_num, col_id("error"),
                row_num, col_id("error"),
                {
                    'type': 'formula',
                    'criteria': f'=${col_letter("error")}{row_num + 1}<>""',
                    'format': workbook.add_format({'bg_color': 'yellow'})
                }
            )

            # Color automatic_extraction dark red if empty or ambiguous
            worksheet.conditional_format(
                row_num, col_id("automatic_extraction"),
                row_num, col_id("automatic_extraction"),
                {
                    'type': 'formula',
                    'criteria': f'=AND(${col_letter("answer_sentences")}{row_num + 1}<>"", OR(${col_letter("automatic_extraction")}{row_num + 1}="", ISNUMBER(SEARCH("#", ${col_letter("automatic_extraction")}{row_num + 1}))))',
                    'format': workbook.add_format({'bg_color': 'red'})
                }
            )

            if not samples:
                # Color trace_label_correct red if automatic and manual extraction are wrong
                worksheet.conditional_format(
                    row_num, col_id("trace_label_correct"),
                    row_num, col_id("trace_label_correct"),
                    {
                        'type': 'formula',
                        'criteria': f'=AND(${col_letter("automatic_extraction")}{row_num + 1}<>${col_letter("correct_answer")}{row_num + 1},${col_letter("manual_extraction")}{row_num + 1}<>${col_letter("correct_answer")}{row_num + 1})',
                        'format': workbook.add_format({'bg_color': pastel_red})
                    }
                )

                # Color trace_label_correct green if automatic or manual extraction are right
                worksheet.conditional_format(
                    row_num, col_id("trace_label_correct"),
                    row_num, col_id("trace_label_correct"),
                    {
                        'type': 'formula',
                        'criteria': f'=OR(${col_letter("automatic_extraction")}{row_num + 1}=${col_letter("correct_answer")}{row_num + 1}, ${col_letter("manual_extraction")}{row_num + 1}=${col_letter("correct_answer")}{row_num + 1})',
                        'format': workbook.add_format({'bg_color': pastel_green})
                    }
                )

                # Color manual extraction dark red if it is necessary (automatic extraction failed (empty or undecisive))
                worksheet.conditional_format(
                    row_num, col_id("manual_extraction"),
                    row_num, col_id("manual_extraction"),
                    {
                        'type': 'formula',
                        'criteria': f'=AND(${col_letter("manual_extraction")}{row_num + 1}="", OR(${col_letter("automatic_extraction")}{row_num + 1}="", ISNUMBER(SEARCH("#", ${col_letter("automatic_extraction")}{row_num + 1}))))',
                        'format': workbook.add_format({'bg_color': 'red'})
                    }
                )

                # Color model_choice_correct red if it is wrong
                worksheet.conditional_format(
                    row_num, col_id("model_choice_correct"),
                    row_num, col_id("model_choice_correct"),
                    {
                        'type': 'formula',
                        'criteria': f'=${col_letter("model_choice")}{row_num + 1}<>${col_letter("correct_answer")}{row_num + 1}',
                        'format': workbook.add_format({'bg_color': pastel_red})
                    }
                )

                # Color the model_choice_correct green if it is right
                worksheet.conditional_format(
                    row_num, col_id("model_choice_correct"),
                    row_num, col_id("model_choice_correct"),
                    {
                        'type': 'formula',
                        'criteria': f'=${col_letter("model_choice")}{row_num + 1}=${col_letter("correct_answer")}{row_num + 1}',
                        'format': workbook.add_format({'bg_color': pastel_green})
                    }
                )

                # Color labels_match green if they match
                worksheet.conditional_format(
                    row_num, col_id("labels_match"),
                    row_num, col_id("labels_match"),
                    {
                        'type': 'formula',
                        'criteria': f'=OR(${col_letter("automatic_extraction")}{row_num + 1}=${col_letter("model_choice")}{row_num + 1}, ${col_letter("manual_extraction")}{row_num + 1}=${col_letter("model_choice")}{row_num + 1})',
                        'format': workbook.add_format({'bg_color': 'green'})
                    }
                )

                # Color labels_do_match red if they do not match
                worksheet.conditional_format(
                    row_num, col_id("labels_match"),
                    row_num, col_id("labels_match"),
                    {
                        'type': 'formula',
                        'criteria': f'=AND(${col_letter("automatic_extraction")}{row_num + 1}<>${col_letter("model_choice")}{row_num + 1}, ${col_letter("manual_extraction")}{row_num + 1}<>${col_letter("model_choice")}{row_num + 1})',
                        'format': workbook.add_format({'bg_color': 'red'})
                    }
                )

                # Color baseline_model_choice_correct red if it is wrong
                worksheet.conditional_format(
                    row_num, col_id("baseline_model_choice_correct"),
                    row_num, col_id("baseline_model_choice_correct"),
                    {
                        'type': 'formula',
                        'criteria': f'=${col_letter("baseline_model_choice")}{row_num + 1}<>${col_letter("correct_answer")}{row_num + 1}',
                        'format': workbook.add_format({'bg_color': pastel_red})
                    }
                )

                # Color baseline_model_choice_correct green if it is right
                worksheet.conditional_format(
                    row_num, col_id("baseline_model_choice_correct"),
                    row_num, col_id("baseline_model_choice_correct"),
                    {
                        'type': 'formula',
                        'criteria': f'=${col_letter("baseline_model_choice")}{row_num + 1}=${col_letter("correct_answer")}{row_num + 1}',
                        'format': workbook.add_format({'bg_color': pastel_green})
                    }
                )

                # Color runs_match green if they match
                worksheet.conditional_format(
                    row_num, col_id("runs_match"),
                    row_num, col_id("runs_match"),
                    {
                        'type': 'formula',
                        'criteria': f'=${col_letter("runs_match")}{row_num + 1}=1',
                        'format': workbook.add_format({'bg_color': 'green'})
                    }
                )

                # Color runs_match red if they do not match
                worksheet.conditional_format(
                    row_num, col_id("runs_match"),
                    row_num, col_id("runs_match"),
                    {
                        'type': 'formula',
                        'criteria': f'=${col_letter("runs_match")}{row_num + 1}=0',
                        'format': workbook.add_format({'bg_color': 'red'})
                    }
                )

            else:
                # Color correct_classification orange if filled in
                worksheet.conditional_format(
                    row_num, col_id("correct_classification"),
                    row_num, col_id("correct_classification"),
                    {
                        'type': 'formula',
                        'criteria': f'=${col_letter("correct_classification")}{row_num + 1}="1"',
                        'format': workbook.add_format({'bg_color': 'green'})
                    }
                )

                # Color correct_classification orange if filled in
                worksheet.conditional_format(
                    row_num, col_id("correct_classification"),
                    row_num, col_id("correct_classification"),
                    {
                        'type': 'formula',
                        'criteria': f'=AND(${col_letter("correct_classification")}{row_num + 1}<>"1", ${col_letter("correct_classification")}{row_num + 1}<>"")',
                        'format': workbook.add_format({'bg_color': 'red'})
                    }
                )

        workbook.close()

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
    def analyze_trace_label_match(test_result, baseline_test_result = None):
        single_results: List[SingleResult] = TraceAnalyzer.get_single_results(test_result)

        baseline_model_choices: List[str] = []
        baseline_uuid = ""
        if baseline_test_result:
            baseline_model_choices = TraceAnalyzer.get_model_choices(baseline_test_result)
            baseline_uuid = baseline_test_result["uuid"]

        formatted_single_results = [
            {
                "question_id": single_result.data["question_id"],
                "question": single_result.data["question"],
                "last_prompt": single_result.data["last_prompt"],
                "labels": single_result.data["labels"],
                "reasoning": single_result.data["completions"][0]["reasoning"]["text"].strip(),
                "model_choice": single_result.data["model_choice"],
                "correct_answer": single_result.data["correct_answer"],
                "is_correct": 1 if single_result.data["is_correct"] else 0,
            } for single_result in single_results
        ]
        model_name = test_result["model"]
        cot_uuid = test_result["uuid"]

        unresolved_traces = []
        ambiguous_traces = []
        extractable_traces = []
        question_command_traces = []

        table_rows = []

        nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        answer_sentence_indicators = []
        definite_answer_sentence_indicators = []
        after_answer_sentence_indicators = []
        exclusion_indicators = []

        #if model_name == "llama-2-7b-chat" or model_name == "llama-2-13b-chat":
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
            #r"is correct",
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
        ]

        question_command_indicators = [
            rf"What is [{re.escape(in_sentence_chars)}]+ answer?",
            rf"What do you think?",
            rf"Please [{re.escape(in_sentence_chars)}]+.",
            rf"Which answer [{re.escape(in_sentence_chars)}]+?",
            rf"Can you [{re.escape(in_sentence_chars)}]+?"
        ]

        if "orca" in model_name:
            tokenizer._params.abbrev_types.remove({'a', 'b', 'c', 'd'})
            definite_answer_sentence_indicators.append("### Final Answer")

        for result_id, formatted_single_result in enumerate(formatted_single_results):
            split_paragraphs = True

            if split_paragraphs:
                # Step 1: Split the text by double linebreaks
                paragraphs = formatted_single_result["reasoning"].split('\n\n')

                # Step 2: Merge paragraphs that belong together
                merged_paragraphs = []
                i = 0
                while i < len(paragraphs):
                    if paragraphs[i].endswith(':') and i + 1 < len(paragraphs):
                        # Merge current sentence with the next one
                        merged_paragraphs.append(paragraphs[i] + ' ' + paragraphs[i + 1])
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
                trace_sentences = tokenizer.tokenize(formatted_single_result["reasoning"])

            trace_answer_sentences = []
            question_command_sentences = []

            # Loop over all sentences of the reasoning trace
            for trace_sentence_id, trace_sentence in enumerate(trace_sentences):
                # Skip sentences which are most likely not relevant
                if any(re.search(exclusion_indicator, trace_sentence) for exclusion_indicator in exclusion_indicators):
                    continue

                # Check for matches with after_answer_sentence_indicator
                if any(after_answer_sentence_indicator in trace_sentence for after_answer_sentence_indicator in after_answer_sentence_indicators):  # Include sentence before
                    if len(TraceAnalyzer.extract_label_matches_from_sentence(trace_sentence, formatted_single_result["labels"])) == 0:  # Indicator stands alone, not e.g. "Option (B) improve existing products: This option is correct!"
                        trace_answer_sentences.append(trace_sentences[trace_sentence_id-1] + "\n" + trace_sentence)
                        continue

                # If sentence is not skipped until here, search for answer indicators
                if any(re.search(answer_sentence_indicator, trace_sentence) for answer_sentence_indicator in answer_sentence_indicators):
                    trace_answer_sentences.append(trace_sentence)

                # The last sentence contains a question or a command, so the model probably did not chose a label
                if trace_sentence_id == len(trace_sentences) - 1 and any(re.search(question_command_indicator, trace_sentence) for question_command_indicator in question_command_indicators):
                    question_command_sentences.append(trace_sentence)

            for trace_answer_sentence in trace_answer_sentences:
                if any(re.search(definite_answer_sentence_indicator, trace_answer_sentence) for definite_answer_sentence_indicator in definite_answer_sentence_indicators):
                    trace_answer_sentences = [trace_answer_sentence]
                    break

            table_row = {
                "question_id": formatted_single_result["question_id"],
                "question": formatted_single_result["question"],
                "answer_options": re.search("\n\nAnswer Choices:\n(.*)\n\nAmong", formatted_single_result["last_prompt"], re.DOTALL).group(1),
                "correct_answer": formatted_single_result["correct_answer"],

                "reasoning": formatted_single_result["reasoning"],
                "answer_sentences": "",
                "automatic_extraction": "",
                "manual_extraction": "",
                "error": "",
                "trace_label_correct": "",

                "model_choice": formatted_single_result["model_choice"],
                "model_choice_correct": formatted_single_result["is_correct"],
                "labels_match": "",

                "baseline_model_choice": baseline_model_choices[result_id] if baseline_test_result else "",
                "baseline_model_choice_correct": (1 if baseline_model_choices[result_id] == formatted_single_result["correct_answer"] else 0) if baseline_test_result else "",
                "runs_match": "",

                "comment": ""
            }

            # At least one answer sentence was found
            if len(trace_answer_sentences) > 0:
                table_row.update(answer_sentences="\n======\n".join(trace_answer_sentences))

                extracted_labels = []
                for trace_answer_sentence in trace_answer_sentences:
                    extracted_labels += TraceAnalyzer.extract_label_matches_from_sentence(trace_answer_sentence, formatted_single_result["labels"])

                extracted_labels = sorted(set(extracted_labels))

                if len(extracted_labels) > 1:
                    table_row.update(error=Error.AMBIGUOUS.value)
                    ambiguous_traces.append([trace_sentences, trace_answer_sentences, list(extracted_labels), formatted_single_result["correct_answer"]])

                table_row.update(automatic_extraction="\n======\n".join(extracted_labels))
                extractable_traces.append([trace_answer_sentences, extracted_labels])

            elif len(question_command_sentences) > 0:
                table_row.update(error=Error.QUESTION.value)
                question_command_traces.append([formatted_single_result["question_id"], trace_sentences, question_command_sentences])

            else:
                table_row.update(answer_sentences="")
                unresolved_traces.append([formatted_single_result["question_id"], trace_sentences])

            table_rows.append(table_row)

        print(json.dumps(unresolved_traces, indent=2))
        print(f"Traces containing an extractable answer: {len(extractable_traces)}. \n-> {len(ambiguous_traces)} of them are ambiguous.")
        print(f"Traces ending with a question or a command: {len(question_command_traces)}")
        print(f"Unresolved traces: {len(unresolved_traces)}")
        # print(json.dumps(unresolved_traces, indent=2))
        # print(json.dumps(extractable_traces, indent=2))

        analysis = {
            "cot_uuid": cot_uuid,
            "baseline_uuid": baseline_uuid,
            "model_name": model_name,
            "result_rows": table_rows
        }

        return analysis

    method_mapping: Dict[str, callable] = {
        "trace-label-match": analyze_trace_label_match
    }


if __name__ == '__main__':
    fire.Fire(TraceAnalyzer.analyze)