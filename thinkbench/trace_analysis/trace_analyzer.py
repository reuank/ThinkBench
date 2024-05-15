import json
import datetime
from string import Template

import nltk.data
import xlsxwriter
from pathlib import Path
from typing import List, Dict

import fire


class SingleResult:
    def __init__(self, data):
        self.data = data


class TraceAnalyzer:
    @staticmethod
    def analyze(method: str, path: str):
        path: Path = Path(path)
        files: List[Path] = []

        nltk.download('punkt')

        if path.is_dir():
            p = path.glob('**/*')
            files = [x for x in p if x.is_file()]
        elif path.is_file():
            files = [path]
        elif path is None:
            raise ValueError("The provided path is invalid.")

        test_results = [TraceAnalyzer.load_result_file(file) for file in files]

        analysis_results = [TraceAnalyzer.method_mapping[method](test_result) for test_result in test_results]

        #TraceAnalyzer.write_to_json_file(analysis_results, "test")
        TraceAnalyzer.write_to_xlsx(analysis_results)

    @staticmethod
    def load_result_file(file: Path):
        file = open(file)
        data = json.load(file)

        return data

    @staticmethod
    def get_single_results(data: Dict) -> List[SingleResult]:
        return [SingleResult(single_result) for single_result in data["results"]]

    @staticmethod
    def write_to_json_file(data, suffix: str = ""):
        filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{suffix}.json"
        f = open(filename, "a")
        f.write(json.dumps(data, indent=2))

        print(f"File {filename} written.")

    @staticmethod
    def write_to_xlsx(analysis_results: List[Dict]):
        pastel_red = "#FFCCCC"
        pastel_green = "#CCFFCC"

        workbook = xlsxwriter.Workbook("test.xlsx")
        for analysis_result in analysis_results:
            model_name = analysis_result["model_name"]
            worksheet = workbook.add_worksheet(model_name)

            cell_format = workbook.add_format({
                "valign": "top",
                "text_wrap": True
            })

            for col_num, col_data in enumerate(analysis_result["result_rows"][0].keys()):
                worksheet.set_column(col_num + 1, col_num + 1, 20)

            worksheet.set_column('C:C', 60)  # reasoning
            worksheet.set_column('E:E', 60)  # answer sentence

            # Table content
            for row_num, row_data in enumerate(analysis_result["result_rows"]):
                for col_num, cell_data in enumerate(list(row_data.values())):
                    worksheet.write(row_num + 2, col_num + 1, cell_data, cell_format)

            # Define the table range (start_row, start_col, end_row, end_col)
            start_row = 1
            start_col = 1
            end_row = len(analysis_result["result_rows"]) + 1
            end_col = len(analysis_result["result_rows"][0].values())

            # Add the table with headers
            worksheet.add_table(start_row, start_col, end_row, end_col, {'columns': [{'header': col} for col in list(analysis_result["result_rows"][0].keys())]})

            for row_num in range(start_row + 1, end_row + 1):  # Skip header row
                # Color automatic extraction red if it is wrong
                worksheet.conditional_format(
                    row_num, 4,  # Column E
                    row_num, 4,
                    {
                        'type': 'formula',
                        'criteria': f'=AND($E{row_num + 1}<>$I{row_num + 1},$E{row_num + 1}<>"")',
                        'format': workbook.add_format({'bg_color': pastel_red})
                    }
                )

                # Color automatic extraction dark red if it is empty
                worksheet.conditional_format(
                    row_num, 4,  # Column E
                    row_num, 4,
                    {
                        'type': 'formula',
                        'criteria': f'=$E{row_num + 1}=""',
                        'format': workbook.add_format({'bg_color': 'red'})
                    }
                )

                # Color automatic extraction green if it is right
                worksheet.conditional_format(
                    row_num, 4,  # Column E
                    row_num, 4,
                    {
                        'type': 'formula',
                        'criteria': f'=$E{row_num + 1}=$I{row_num + 1}',
                        'format': workbook.add_format({'bg_color': pastel_green})
                    }
                )

                # Color manual extraction red if it is wrong
                worksheet.conditional_format(
                    row_num, 5,  # Column F
                    row_num, 5,
                    {
                        'type': 'formula',
                        'criteria': f'=$F{row_num + 1}<>$I{row_num + 1}',
                        'format': workbook.add_format({'bg_color': pastel_red})
                    }
                )

                # Color manual extraction green if it is right
                worksheet.conditional_format(
                    row_num, 5,  # Column F
                    row_num, 5,
                    {
                        'type': 'formula',
                        'criteria': f'=$F{row_num + 1}=$I{row_num + 1}',
                        'format': workbook.add_format({'bg_color': pastel_green})
                    }
                )

                # Color the model choice red if it is wrong
                worksheet.conditional_format(
                    row_num, 6,  # Column G
                    row_num, 6,
                    {
                        'type': 'formula',
                        'criteria': f'=$G{row_num + 1}<>$I{row_num + 1}',
                        'format': workbook.add_format({'bg_color': pastel_red})
                    }
                )

                # Color the model choice red if it is right
                worksheet.conditional_format(
                    row_num, 6,  # Column G
                    row_num, 6,
                    {
                        'type': 'formula',
                        'criteria': f'=$G{row_num + 1}=$I{row_num + 1}',
                        'format': workbook.add_format({'bg_color': pastel_green})
                    }
                )

        workbook.close()

    @staticmethod
    def extract_label_matches(sentence: str, labels: List[str]) -> List[str]:
        remove_patterns = [
            Template("among ${first_label} through ${last_label}")
        ]
        label_patterns = [
            Template("(${label})"),
            Template("${label})"),
            Template(" ${label} "),
            Template("option ${label}"),
            Template("Option ${label}"),
        ]

        matches = {}

        for label in labels:
            for remove_pattern in remove_patterns:
                sentence.replace(remove_pattern.substitute(
                    first_label=labels[0],
                    last_label=labels[-1],
                ), "")
            for label_pattern in label_patterns:
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
    def analyze_trace_label_match(test_result):
        single_results: List[SingleResult] = TraceAnalyzer.get_single_results(test_result)

        formatted_single_results = [
            {
                "question_id": single_result.data["question_id"],
                "labels": single_result.data["labels"],
                "reasoning": single_result.data["completions"][0]["reasoning"]["text"],
                "model_choice": single_result.data["model_choice"],
                "correct_answer": single_result.data["correct_answer"],
            } for single_result in single_results
        ]
        model_name = test_result["model"]

        unresolved_traces = []
        extractable_traces = []

        table_rows = []

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        answer_sentence_indicators = []

        if model_name == "llama-2-7b-chat":
            answer_sentence_indicators = [
                "the correct answer is",
                "The correct answer is",
                "is the correct answer",
                "the best answer",
                "The best answer",
                "the answer is",
                "The answer is"
            ]

        for formatted_single_result in formatted_single_results:
            table_row = {
                "question_id": formatted_single_result["question_id"],
                "reasoning": formatted_single_result["reasoning"],
                #"is_extractable": "",
                "answer_sentences": "",
                "automatic_extraction": "",
                "manual_extraction": "",
                "model_choice": formatted_single_result["model_choice"],
                "labels_do_match": "",
                "correct_answer": formatted_single_result["correct_answer"],
                "labels_match_and_correct": ""
            }

            trace_sentences = tokenizer.tokenize(formatted_single_result["reasoning"])
            trace_answer_sentences = []

            # Loop over all sentences of the reasoning trace
            for trace_sentence_id, trace_sentence in enumerate(trace_sentences):
                # Check for matches with answer_sentence_indicators
                if any(answer_sentence_indicator in trace_sentence for answer_sentence_indicator in answer_sentence_indicators):
                    trace_answer_sentences.append(trace_sentence)

            # At least one answer sentence was found
            if len(trace_answer_sentences) > 0:
                # table_row.update(is_extractable=True)
                table_row.update(answer_sentences="\n###\n".join(trace_answer_sentences))

                extracted_labels = []
                for trace_answer_sentence in trace_answer_sentences:
                    extracted_labels += TraceAnalyzer.extract_label_matches(trace_answer_sentence, formatted_single_result["labels"])

                extracted_labels = set(extracted_labels)

                table_row.update(automatic_extraction="\n###\n".join(extracted_labels))

                extractable_traces.append([trace_answer_sentences, extracted_labels])
            else:
                # table_row.update(is_extractable=False)
                table_row.update(answer_sentences="")

                unresolved_traces.append([trace_answer_sentences])

            table_rows.append(table_row)

        print(f"Traces containing an extractable answer: {len(extractable_traces)}")
        print(f"Traces unresolved: {len(unresolved_traces)}")

        return {
            "model_name": model_name,
            "result_rows": table_rows
        }

    method_mapping: Dict[str, callable] = {
        "trace-label-match": analyze_trace_label_match
    }


if __name__ == '__main__':
    fire.Fire(TraceAnalyzer.analyze)