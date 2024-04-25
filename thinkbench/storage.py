import json
import os
from abc import ABC, abstractmethod
import datetime
from pathlib import Path

from numpy import float32

from completion import CompletionHistory
from testcase import TestCaseResult


class TotalResultEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float32):
            return float(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        else:
            return obj.__dict__


class StorageBackend(ABC):
    @abstractmethod
    def store(self, test_case_result: TestCaseResult):
        raise NotImplementedError


class JsonFileStorage(StorageBackend):
    def __init__(self):
        try:
            self.hostname = os.environ.get("TB_HOSTNAME")
            self.output_path = os.environ.get("TB_OUTPUT_PATH")
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            if not self.hostname or not self.output_path:
                raise KeyError
        except KeyError:
            print("Please specify an output path and a hostname.")
            exit()

    def store(self, test_case_result: TestCaseResult):
        filename = f"{self.output_path}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{test_case_result['benchmark_name']}_{test_case_result['model']}_{test_case_result['dataset_name']}-{test_case_result['metrics']['total_results']}_labels-{test_case_result['label_numbering']}_{'use-chat-template' if test_case_result['use_chat_template'] else 'no-chat-template'}_{test_case_result['inference_backend']}_{test_case_result['hostname']}.json"
        f = open(filename, "a")
        f.write(json.dumps(test_case_result, cls=TotalResultEncoder, indent=4))
        f.close()

        print(f"File {filename} written.")