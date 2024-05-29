from pathlib import Path

LIBRARY_ROOT: Path = Path(__file__).parent

# Server options
SERVER_HOST: str = "localhost"

# Logging options
LOG_INFO: bool = True
LOG_ERROR: bool = True
TIMER_VERBOSE: bool = False
INFERENCE_BACKEND_VERBOSE: bool = False
PRINT_SEPARATOR_LENGTH: int = 45
PRINT_SEPARATOR: str = "="

# Dataset defaults
RANDOM_DATA_SAMPLES_SEED: int = 1337

# Model config defaults
GGUF_FILE_EXTENSION: str = "Q4_K_M.gguf"
DEFAULT_MODEL_FILENAME: str = "model.safetensors"

# Completion defaults
COMPLETION_SEED: int = 1234
N_GPU_LAYERS: int = 1000
LABEL_MAX_LOGPROBS: int = 50
REASONING_MAX_LOGPROBS: int = 1
REASONING_MAX_TOKENS: int = 2048
TOKENIZE_BEFORE: bool = True

# Decoder defaults
GREEDY_DECODER_TEMPERATURE: float = 0.0
DEFAULT_DECODER_REPEAT_PENALTY: float = 1.0
DEFAULT_DECODER_REPEAT_LAST_N: int = 64
DEFAULT_DECODER_MIN_P: float = 0.0
DEFAULT_DECODER_TOP_P: float = 1.0
DEFAULT_DECODER_TOP_K: int = 100

# Default prompt templates
DEFAULT_OPTIONAL_CONTEXT_TEMPLATE: str = (
    "{% if single_data_instance.context %}"
    "Passage:\n"
    "{{ single_data_instance.context }}"
    "\n\n"
    "{% endif %}"
)
DEFAULT_QUESTION_TEMPLATE: str = (
    "Question:\n"
    "{{ single_data_instance.question }}"
    "\n\n"
)
DEFAULT_ANSWER_OPTION_TEMPLATE: str = (
    "Answer Choices:\n"
    "{% for label in single_data_instance.answer_labels %}"
    "({{ label }}) {{ single_data_instance.answer_texts[loop.index0] }}{{ '\n' if not loop.last }}"
    "{% endfor %}"
    "\n\n"
)
