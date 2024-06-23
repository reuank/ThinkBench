import curses
import textwrap
from typing import List

from evaluation.classification.classification_result import ClassificationResult
from evaluation.classification.trace_class import TraceClass
from utils.logger import Logger


class InteractiveClassifier:
    def __init__(self):
        self.classification_results: List[ClassificationResult] = []
        self.trace_classes: List[(int, str)] = [[trace_class.value, trace_class.name] for trace_class in TraceClass]

    def main(self, standard_screen):
        curses.curs_set(0)  # Hide the cursor
        standard_screen.keypad(True)  # Enable keypad mode
        all_rows = [classification_result["results"] for classification_result in self.classification_results]
        total_samples = sum(len(row) for row in all_rows)
        total_sample_counter = 0

        for classification_result_id, classification_result in enumerate(self.classification_results):
            for single_classification_id, single_classification in enumerate(classification_result["results"]):
                total_sample_counter += 1
                display_text = f"{Logger.print_header('REASONING', False)}\n" \
                               f"{single_classification['reasoning'].strip()}\n\n"

                display_text += f"{Logger.print_header('COT MODEL CHOICE', False)}\n" \
                                f"{single_classification['cot_model_choice']}\n\n"

                display_text += f"{Logger.print_header('TRACE CLASSES', False)}\n"
                for trace_class in self.trace_classes:
                    display_text += f"({trace_class[0]}) {trace_class[1].replace('_', ' ')}\n"

                trace_class_id = ""
                while not TraceClass.is_trace_class(trace_class_id):
                    trace_class_id = InteractiveClassifier.classify_text(standard_screen, display_text, total_sample_counter, total_samples)

                self.classification_results[classification_result_id]["results"][single_classification_id]["manual_class_id"] = int(trace_class_id)

    @staticmethod
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

        # pad.refresh(pad_pos, 0, 0, 0, max_y - 2, max_x - 1)
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
            elif chr(key).isdigit() and TraceClass.is_trace_class(chr(key)):
                return chr(key)

    def classify(self, classification_results: List[ClassificationResult]):
        self.classification_results = classification_results

        try:
            curses.wrapper(self.main)
        except KeyboardInterrupt:
            pass
