import curses
import random
import textwrap
from pathlib import Path
from typing import List, Dict, Union

from constants import TRACE_SAMPLES_PER_RUN
from storage.backends.csv_file_storage import CsvFileStorage
from trace_analysis.automatic_trace_classifier import AutomaticTraceClassifier, TraceClass
from trace_analysis.trace_classifier import TraceClassifier
from utils.cli_interactions import Interaction
from utils.logger import Logger


class ManualTraceClassifier(TraceClassifier):
    @staticmethod
    def store_trace_samples(cot_results: List[Dict], non_cot_results: List[Dict], interactive: bool, override: bool):
        csv_file_storage: CsvFileStorage = CsvFileStorage()
        all_samples: Dict = {}
        sampled_models = []

        for cot_result_id, cot_result in enumerate(cot_results):
            sample_again = True
            non_cot_result = non_cot_results[cot_result_id]

            samples_file_name = csv_file_storage.get_samples_file_name(
                model_name=cot_result["model"],
                cot_uuid=cot_result["uuid"],
                non_cot_uuid=non_cot_result["uuid"],
            )

            non_cot_model_choices = [single_result["model_choice"] for single_result in non_cot_result["results"]]
            cot_model_choices = [single_result["model_choice"] for single_result in cot_result["results"]]

            samples_file_path: Path = csv_file_storage.analysis_path / samples_file_name
            if samples_file_path.is_file() and not override:
                # TODO: Add possibility to continue labeling process
                manual_classification_file_rows = csv_file_storage.load_analysis_result(samples_file_path)
                manual_class_ids = [
                    0 if manual_classification_row["manual_class_id"] == ""
                    else int(manual_classification_row["manual_class_id"])
                    for manual_classification_row in manual_classification_file_rows
                ]

                num_labeled = sum(
                    manual_class_id != 0 and manual_class_id != ""
                    for manual_class_id in manual_class_ids
                )

                sample_again = Interaction.query_yes_no(
                    question=f"A samples file {samples_file_name} already exists for model {cot_result['model']}."
                             f"\nIt contains {num_labeled} labeled traces. Do you want to continue?",
                    default="no"
                )

            if sample_again:
                cot_result_with_samples = ManualTraceClassifier.add_samples_to_result(
                    result=cot_result,
                    seed=cot_result["model"]
                )

                sample_file_rows = []

                for cot_sample_result_row in cot_result_with_samples["sample_results"]:
                    if "reasoning" not in cot_sample_result_row["completions"][0].keys():
                        raise ValueError("CoT row does not contain a reasoning trace.")

                    runs_match = non_cot_model_choices[cot_sample_result_row["question_id"]] == cot_model_choices[cot_sample_result_row["question_id"]]
                    model_choice = cot_model_choices[cot_sample_result_row["question_id"]]

                    sample_file_rows.append({
                        "cot_uuid": cot_result["uuid"],
                        "non_cot_uuid": non_cot_result["uuid"],
                        "question_id": cot_sample_result_row["question_id"],
                        "reasoning": cot_sample_result_row["completions"][0]["reasoning"]["text"].strip(),
                        "model_choice": model_choice,
                        # "runs_match": runs_match,
                        "manual_class_id": 0
                    })

                all_samples.update(
                    {
                        cot_result["model"]: {
                            "sample_file_rows": sample_file_rows,
                            "file_name": samples_file_name
                        }
                    }
                )

        if interactive:
            all_samples = ManualTraceClassifier.display_and_classify_texts(all_samples)

        for model, model_data in all_samples.items():
            sampled_models.append(model)
            csv_file_storage.store_analysis_result(
                headers=["cot_uuid", "non_cot_uuid", "question_id", "reasoning", "model_choice", "manual_class_id"],
                rows=[list(sample_row.values()) for sample_row in model_data["sample_file_rows"]],
                file_name=model_data["file_name"]
            )

        if len(all_samples) > 0:
            Logger.info(f"Manual classification files for models {', '.join(sampled_models)} were written to {csv_file_storage.analysis_path}.")

    @staticmethod
    def display_and_classify_texts(
            all_samples: Dict[str, Dict[str, Union[str, List[Dict]]]]
    ) -> Dict[str, Dict[str, Union[str, List[Dict]]]]:
        trace_classes = [[trace_class.value, trace_class.name] for trace_class in TraceClass]

        def classify_text(standard_screen, text, sample_index, total_samples):
            max_y, max_x = standard_screen.getmaxyx()

            wrapped_text = []
            for line in text.split("\n"):
                wrapped_lines = textwrap.wrap(line, width=max_x)
                if not wrapped_lines:
                    wrapped_text.append("")  # Preserve blank lines
                else:
                    wrapped_text.extend(wrapped_lines)

            pad = curses.newpad(max(len(wrapped_text), max_y), max_x)
            pad_pos = 0

            # Write the text to the pad
            for i, line in enumerate(wrapped_text):
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
                    f"– Enter a trace class id (1-4), scroll (UP/DOWN), or abort (ctrl+c): ")
                standard_screen.refresh()

                key = standard_screen.getch()

                # Handle scrolling
                if key == curses.KEY_UP and pad_pos > 0:
                    pad_pos -= 1
                elif key == curses.KEY_DOWN and pad_pos < len(wrapped_text) - max_y + 2:
                    pad_pos += 1
                elif chr(key).isdigit() and AutomaticTraceClassifier.is_trace_class(chr(key)):
                    return chr(key)

        def main(standard_screen):
            curses.curs_set(0)  # Hide the cursor
            standard_screen.keypad(True)  # Enable keypad mode
            all_sample_rows = [model_data["sample_file_rows"] for model, model_data in all_samples.items()]
            total_samples = sum(len(sample_rows) for sample_rows in all_sample_rows)
            total_sample_counter = 0
            for model, model_data in all_samples.items():
                for sample_row_index, sample_row in enumerate(model_data["sample_file_rows"]):
                    total_sample_counter += 1
                    display_text = f"{Logger.print_header('REASONING', False)}\n" \
                                   f"{sample_row['reasoning'].strip()}\n\n"

                    display_text += f"{Logger.print_header('COT MODEL CHOICE', False)}\n" \
                                    f"{sample_row['model_choice']}\n\n"

                    display_text += f"{Logger.print_header('TRACE CLASSES', False)}\n"
                    for trace_class in trace_classes:
                        display_text += f"({trace_class[0]}) {trace_class[1].replace('_', ' ')}\n"

                    trace_class_id = ""
                    while not AutomaticTraceClassifier.is_trace_class(trace_class_id):
                        trace_class_id = classify_text(standard_screen, display_text, total_sample_counter, total_samples)
                    all_samples[model]["sample_file_rows"][sample_row_index]["manual_class_id"] = trace_class_id

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
