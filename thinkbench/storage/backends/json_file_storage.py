import datetime
import json
import os
from pathlib import Path

from storage.storage_backend import StorageBackend, STORAGE_BACKEND_REGISTRY
from benchmark.results import TestCaseResult
from utils.encoders import TotalResultEncoder


@STORAGE_BACKEND_REGISTRY.register(name="json_file_storage")
class JsonFileStorage(StorageBackend):
    def __init__(self):
        try:
            self.hostname = os.environ.get("TB_HOSTNAME")
            self.output_path = os.environ.get("TB_OUTPUT_PATH")
            if not self.hostname or not self.output_path:
                raise KeyError
            else:
                Path(self.output_path).mkdir(parents=True, exist_ok=True)
        except KeyError:
            print("Please specify an output path and a hostname. Did you forget to source .env?")
            exit()

    def store(self, test_case_result: TestCaseResult):
        filename = f"{self.output_path}/" \
                   f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" \
                   f"_{test_case_result['benchmark_name']}" \
                   f"_{test_case_result['model']}" \
                   f"_{test_case_result['dataset_name']}-{test_case_result['metrics']['total_results']}" \
                   f"_labels-{test_case_result['label_numbering']}" \
                   f"_{'use-chat-template' if test_case_result['use_chat_template'] else 'no-chat-template'}" \
                   f"_{test_case_result['inference_backend']}" \
                   f"_{test_case_result['hostname']}.json"
        f = open(filename, "a")
        f.write(json.dumps(test_case_result, cls=TotalResultEncoder, indent=2))
        f.close()

        print(f"File {filename} written.")
