import random
from typing import List, Dict

from constants import TRACE_SAMPLES_PER_RUN
from storage.backends.csv_file_storage import CsvFileStorage
from utils.logger import Logger


class TraceSamplesStorer:
    @staticmethod
    def store_trace_samples(cot_results: List[Dict], non_cot_results: List[Dict]):
        csv_file_storage: CsvFileStorage = CsvFileStorage()

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

                sample_rows.append([
                    cot_sample_result_row["question_id"],
                    # cot_sample_result_row["question"],
                    # re.search("\n\nAnswer Choices:\n(.*)\n\nAmong", cot_sample_result_row["last_prompt"], re.DOTALL).group(1),
                    cot_sample_result_row["completions"][0]["reasoning"]["text"].strip(),
                    labels_match,
                    ""
                ])

            samples_file_name = csv_file_storage.get_samples_filename(
                model_name=cot_result['model'],
                cot_uuid=cot_result['uuid'][:8],
                non_cot_uuid=non_cot_result['uuid'][:8]
            )

            csv_file_storage.store_analysis_result(
                headers=["question_id", "reasoning", "labels_match", "manual_category_id"],
                rows=sample_rows,
                filename=samples_file_name
            )

        Logger.info(f"Sample files were written to {csv_file_storage.analysis_path}")

    @staticmethod
    def add_samples_to_result(result: Dict, n_samples: int = TRACE_SAMPLES_PER_RUN, seed: str = "") -> Dict:
        results = result["results"]
        n_results = len(results)

        random.seed(seed)
        samples = [results[random.randint(0, n_results - 1)] for i in range(0, n_samples)]

        assert len(samples) == n_samples

        result.update(sample_results=samples)

        return result
