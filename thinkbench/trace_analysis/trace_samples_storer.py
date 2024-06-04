import curses
import random
from typing import List, Dict

from constants import TRACE_SAMPLES_PER_RUN
from storage.backends.csv_file_storage import CsvFileStorage
from trace_analysis.trace_analysis import Category
from trace_analysis.trace_classifier import TraceClassifier
from utils.logger import Logger


class TraceSamplesStorer:
    @staticmethod
    def store_trace_samples(cot_results: List[Dict], non_cot_results: List[Dict], interactive: bool):
        csv_file_storage: CsvFileStorage = CsvFileStorage()
        all_samples: Dict = {}

        for cot_result_id, cot_result in enumerate(cot_results):
            non_cot_result = non_cot_results[cot_result_id]

            non_cot_model_choices = [single_result["model_choice"] for single_result in non_cot_result["results"]]
            cot_model_choices = [single_result["model_choice"] for single_result in cot_result["results"]]

            cot_result_with_samples = TraceSamplesStorer.add_samples_to_result(
                result=cot_result,
                seed=cot_result["model"]
            )

            sample_rows = []

            for cot_sample_result_row in cot_result_with_samples["sample_results"]:
                if "reasoning" not in cot_sample_result_row["completions"][0].keys():
                    raise ValueError("File does not contain")

                labels_match = non_cot_model_choices[cot_sample_result_row["question_id"]] == cot_model_choices[cot_sample_result_row["question_id"]]

                sample_rows.append({
                    "cot_uuid": cot_result['uuid'],
                    "non_cot_uuid": non_cot_result['uuid'],
                    "question_id": cot_sample_result_row["question_id"],
                    "reasoning": cot_sample_result_row["completions"][0]["reasoning"]["text"].strip(),
                    "labels_match": labels_match,
                    "manual_category_id": ""
                })

            all_samples.update({cot_result["model"]: sample_rows})

        if interactive:
            all_samples = TraceSamplesStorer.display_and_classify_texts(all_samples)

        for model, sample_rows in all_samples.items():
            samples_file_name = csv_file_storage.get_samples_filename(
                model_name=model,
                cot_uuid=sample_rows[0]["cot_uuid"],
                non_cot_uuid=sample_rows[0]["non_cot_uuid"],
            )

            csv_file_storage.store_analysis_result(
                headers=["cot_uuid", "non_cot_uuid", "question_id", "reasoning", "labels_match", "manual_category_id"],
                rows=[list(sample_row.values()) for sample_row in sample_rows],
                filename=samples_file_name
            )

        Logger.info(f"Sample files were written to {csv_file_storage.analysis_path}")


    @staticmethod
    def display_and_classify_texts(all_samples: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        categories = [[category.value, category.name] for category in Category]

        def classify_text(standard_screen, text, sample_index, total_samples):
            max_y, max_x = standard_screen.getmaxyx()

            # Create a new pad
            lines = text.split('\n')
            pad = curses.newpad(max(len(lines), max_y), max_x)
            pad_pos = 0

            # Write the text to the pad
            for i, line in enumerate(lines):
                pad.addstr(i, 0, line)

            #pad.refresh(pad_pos, 0, 0, 0, max_y - 2, max_x - 1)
            standard_screen.refresh()

            # Initial display of the pad content
            while True:
                pad.refresh(pad_pos, 0, 0, 0, max_y - 2, max_x - 1)
                standard_screen.addstr(
                    max_y - 1,
                    0,
                    f"Classify Reasoning Trace {sample_index}/{total_samples} "
                    f"– Enter a category id (1-4), scroll (UP/DOWN), or abort (ctrl+c): ")
                standard_screen.refresh()

                key = standard_screen.getch()

                # Handle scrolling
                if key == curses.KEY_UP and pad_pos > 0:
                    pad_pos -= 1
                elif key == curses.KEY_DOWN and pad_pos < len(lines) - max_y + 2:
                    pad_pos += 1
                elif chr(key).isdigit() and TraceClassifier.is_category(chr(key)):
                    return chr(key)

        def main(standard_screen):
            curses.curs_set(0)  # Hide the cursor
            standard_screen.keypad(True)  # Enable keypad mode
            total_samples = sum(len(sample_rows) for sample_rows in all_samples.values())
            total_sample_counter = 0
            for model, sample_rows in all_samples.items():
                for sample_row_index, sample_row in enumerate(sample_rows):
                    total_sample_counter += 1
                    display_text = f"{Logger.print_header('REASONING', False)}\n" \
                                   f"{sample_row['reasoning'].strip()}\n\n"

                    display_text += f"{Logger.print_header('LABELS MATCH', False)}\n" \
                                    f"{sample_row['labels_match']}\n\n"

                    display_text += f"{Logger.print_header('CATEGORIES', False)}\n"
                    for category in categories:
                        display_text += f"({category[0]}) {category[1]}\n"

                    label_id = ""
                    while not TraceClassifier.is_category(label_id):
                        label_id = classify_text(standard_screen, display_text, total_sample_counter, total_samples)
                    all_samples[model][sample_row_index]["manual_category_id"] = label_id

        try:
            curses.wrapper(main)
        except KeyboardInterrupt:
            pass
        return all_samples

    @staticmethod
    def add_samples_to_result(result: Dict, n_samples: int = TRACE_SAMPLES_PER_RUN, seed: str = "") -> Dict:
        results = result["results"]
        n_results = len(results)

        random.seed(seed)
        samples = [results[random.randint(0, n_results - 1)] for i in range(0, n_samples)]

        assert len(samples) == n_samples

        result.update(sample_results=samples)

        return result
