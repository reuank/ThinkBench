import json
import time
from datetime import datetime
from typing import Dict


class Timer:
    verbose: bool = False
    _instances: Dict[str, "_Timer"] = {}

    @classmethod
    def set_verbosity(cls, verbose: bool):
        cls.verbose = verbose

    class _Timer:
        def __init__(self, name: str, verbose: bool):
            self.time_format: str = "%d.%m.%Y %H:%M:%S"
            self.print_prefix: str = "###"
            self.verbose: bool = verbose
            self.name: str = name
            self.start_time: float = 0.0
            self.end_time: float = 0.0
            self.elapsed_time: float = 0
            self.ran_before: bool = False

        def start_over(self, print_out: bool = False):
            if self.ran_before:
                self.__init__(self.name, self.verbose)
                if self.verbose or print_out:
                    print(f"\n{self.print_prefix} Timer {self.name.upper()} restarted.")
                self.start_over()
            else:
                self.start_time = time.time()
                self.ran_before = True
                if self.verbose or print_out:
                    print(f"{self.print_prefix} {self.name.upper()} timer started at {datetime.fromtimestamp(self.start_time).strftime(self.time_format)}.")

        def end(self, print_out: bool = False):
            if self.ran_before:
                self.end_time = time.time()
                self.elapsed_time = round(self.end_time - self.start_time, 2)
                if self.verbose or print_out:
                    print(f"{self.print_prefix} {self.name.upper()} timer ended at {datetime.fromtimestamp(self.end_time).strftime(self.time_format)} and took {self.elapsed_time} seconds.")

        def to_dict(self):
            return {
                "name": self.name,
                "start_time": datetime.fromtimestamp(self.start_time).strftime(self.time_format),
                "end_time": datetime.fromtimestamp(self.end_time).strftime(self.time_format),
                "elapsed_time": self.elapsed_time,
            }

    @classmethod
    def get_instance(cls, name) -> _Timer:
        if name not in cls._instances:
            cls._instances[name] = cls._Timer(name, cls.verbose)
        return cls._instances[name]

    @classmethod
    def delete_instance(cls, name):
        if name in cls._instances:
            cls._instances.pop(name)

    @classmethod
    def print_instances(cls):
        print(f"### All timers:\n {json.dumps([instance.to_dict() for instance in cls._instances.values()], indent=2)}")
