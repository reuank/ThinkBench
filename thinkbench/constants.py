from pathlib import Path

LIBRARY_ROOT: Path = Path(__file__).parent

# Logging options
TIMER_VERBOSE = False
INFERENCE_BACKEND_VERBOSE = False
PRINT_SEPARATOR_LENGTH = 45
PRINT_SEPARATOR = "="

# Storage options
STORAGE_BACKEND = "json_file_storage"

# Dataset defaults
RANDOM_DATA_SAMPLES_SEED = 1337

# Model config defaults
GGUF_FILE_EXTENSION: str = "Q4_K_M.gguf"
DEFAULT_MODEL_FILENAME: str = "model.safetensors"

# Completion defaults
COMPLETION_SEED: int = 1234
N_GPU_LAYERS: int = 1000

# Decoder defaults
DEFAULT_DECODER_TEMPERATURE: float = 0.0
DEFAULT_DECODER_REPEAT_PENALTY: float = 1.0
DEFAULT_DECODER_REPEAT_LAST_N: int = 64
DEFAULT_DECODER_MIN_P: float = 0.0
DEFAULT_DECODER_TOP_P: float = 1.0
DEFAULT_DECODER_TOP_K: int = 100
