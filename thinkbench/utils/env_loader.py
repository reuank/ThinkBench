import os

from dotenv import load_dotenv

from utils.logger import Logger


class EnvReader:
    _env_cache = {}

    @staticmethod
    def load_env_file(file_path: str):
        load_dotenv(file_path)
        EnvReader._env_cache = {key: os.getenv(key) for key in os.environ}

    @staticmethod
    def get(key: str, default: str = None, required: bool = False) -> str:
        value = EnvReader._env_cache.get(key)
        if default and value is None:
            Logger.info(f"Environment variable {key} is not set, using default value {default}.")
            value = default
        if required and value is None:
            raise ValueError(f"The required environment variable '{key}' is not set and no default value is provided.")
        return value
