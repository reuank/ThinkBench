import json
import os
from abc import ABC, abstractmethod
import datetime

from numpy import float32

from completion_result import NumpyEncoder
from testcase import TestCaseResult


class StorageBackend(ABC):
    @abstractmethod
    def store(self, test_case_result: TestCaseResult):
        raise NotImplementedError


class JsonFileStorage(StorageBackend):
    def __init__(self):
        try:
            self.hostname = os.environ.get("TB_HOSTNAME")
            self.output_path = os.environ.get("TB_OUTPUT_PATH")
            if not self.hostname or not self.output_path:
                raise KeyError
        except KeyError:
            print("Please specify the necessary environment variables.")
            exit()

    def store(self, test_case_result: TestCaseResult):
        filename = f"{self.output_path}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{test_case_result['benchmark_name']}_{test_case_result['model']}_{test_case_result['dataset_name']}-{test_case_result['total_results']}_labels-{test_case_result['label_numbering']}_{'use-chat-template' if test_case_result['use_chat_template'] else 'no-chat-template'}_{test_case_result['inference_backend']}_{test_case_result['hostname']}.json"
        f = open(filename, "a")
        f.write(json.dumps(test_case_result, cls=NumpyEncoder, indent=4))
        f.close()

        print(f"File {filename} written.")