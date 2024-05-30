from constants import GGUF_FILE_EXTENSION, DEFAULT_MODEL_FILENAME
from model_config.hf_model_config import HFModelConfig
from model_config.model_config import QuantizationMethod

from model_config.model_config import MODEL_CONFIG_REGISTRY


MODEL_CONFIG_REGISTRY.register(name="llama-2-7b-chat", flags={"required": True})(HFModelConfig(
    model_name="llama-2-7b-chat",
    chat_template_name="llama-2-chat",
    hf_base_model_repo="meta-llama/Llama-2-7b-chat-hf",
    quantized_model_repos={
        QuantizationMethod.GGUF: ("TheBloke/Llama-2-7B-Chat-GGUF", f"llama-2-7b-chat.{GGUF_FILE_EXTENSION}"),
        QuantizationMethod.GPTQ: ("TheBloke/Llama-2-7B-Chat-GPTQ", DEFAULT_MODEL_FILENAME),
        QuantizationMethod.AWQ: ("TheBloke/Llama-2-7B-Chat-AWQ", DEFAULT_MODEL_FILENAME),
    },
    use_hf_tokenizer=True
))

MODEL_CONFIG_REGISTRY.register(name="llama-2-13b-chat", flags={"required": True})(HFModelConfig(
    model_name="llama-2-13b-chat",
    chat_template_name="llama-2-chat",
    hf_base_model_repo="meta-llama/Llama-2-13b-chat-hf",
    quantized_model_repos={
        QuantizationMethod.GGUF: ("TheBloke/Llama-2-13B-Chat-GGUF", f"llama-2-13b-chat.{GGUF_FILE_EXTENSION}"),
        QuantizationMethod.GPTQ: ("TheBloke/Llama-2-13B-Chat-GPTQ", DEFAULT_MODEL_FILENAME),
        QuantizationMethod.AWQ: ("TheBloke/Llama-2-13B-Chat-AWQ", DEFAULT_MODEL_FILENAME),
    },
    use_hf_tokenizer=True
))


MODEL_CONFIG_REGISTRY.register(name="orca-2-7b", is_default=True, flags={"required": True})(HFModelConfig(
    model_name="orca-2-7b",
    chat_template_name="orca-2",
    hf_base_model_repo="microsoft/Orca-2-7b",
    quantized_model_repos={
        QuantizationMethod.GGUF: ("TheBloke/Orca-2-7B-GGUF", f"orca-2-7b.{GGUF_FILE_EXTENSION}"),
        QuantizationMethod.GPTQ: ("TheBloke/Orca-2-7B-GPTQ", DEFAULT_MODEL_FILENAME),
        QuantizationMethod.AWQ: ("TheBloke/Orca-2-7B-AWQ", DEFAULT_MODEL_FILENAME),
    },
    use_hf_tokenizer=True
))


MODEL_CONFIG_REGISTRY.register(name="orca-2-13b", flags={"required": True})(HFModelConfig(
    model_name="orca-2-13b",
    chat_template_name="orca-2",
    hf_base_model_repo="microsoft/Orca-2-7b",
    quantized_model_repos={
        QuantizationMethod.GGUF: ("TheBloke/Orca-2-13B-GGUF", f"orca-2-13b.{GGUF_FILE_EXTENSION}"),
        QuantizationMethod.GPTQ: ("TheBloke/Orca-2-13B-GPTQ", DEFAULT_MODEL_FILENAME),
        QuantizationMethod.AWQ: ("TheBloke/Orca-2-13B-AWQ", DEFAULT_MODEL_FILENAME),
    },
    use_hf_tokenizer=True
))


MODEL_CONFIG_REGISTRY.register(name="phi-2", flags={"required": True})(HFModelConfig(
    model_name="phi-2",
    chat_template_name="phi-2",
    hf_base_model_repo="microsoft/phi-2",
    quantized_model_repos={
        QuantizationMethod.GGUF: ("TheBloke/Phi-2-GGUF", f"phi-2.{GGUF_FILE_EXTENSION}"),
        QuantizationMethod.GPTQ: ("TheBloke/Phi-2-GPTQ", DEFAULT_MODEL_FILENAME),
        QuantizationMethod.AWQ: ("TheBloke/Phi-2-AWQ", DEFAULT_MODEL_FILENAME),
    },
    use_hf_tokenizer=True
))


MODEL_CONFIG_REGISTRY.register(name="mistral-7b-instruct-v0.2", flags={"required": True})(HFModelConfig(
    model_name="mistral-7b-instruct-v0.2",
    chat_template_name="mistral-7b-instruct",
    hf_base_model_repo="mistralai/Mistral-7B-Instruct-v0.2",
    quantized_model_repos={
        QuantizationMethod.GGUF: ("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", f"mistral-7b-instruct-v0.2.{GGUF_FILE_EXTENSION}"),
        QuantizationMethod.GPTQ: ("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", DEFAULT_MODEL_FILENAME),
        QuantizationMethod.AWQ: ("TheBloke/Mistral-7B-Instruct-v0.2-AWQ", DEFAULT_MODEL_FILENAME),
    },
    use_hf_tokenizer=True
))


MODEL_CONFIG_REGISTRY.register(name="llama-2-70b-chat")(HFModelConfig(
    model_name="llama-2-70b-chat",
    chat_template_name="llama-2-chat",
    hf_base_model_repo="meta-llama/Llama-2-70b-chat-hf",
    quantized_model_repos={
        QuantizationMethod.GGUF: ("TheBloke/Llama-2-70B-Chat-GGUF", f"llama-2-70b-chat.{GGUF_FILE_EXTENSION}"),
        QuantizationMethod.GPTQ: ("TheBloke/Llama-2-70B-Chat-GPTQ", DEFAULT_MODEL_FILENAME),
        QuantizationMethod.AWQ: ("TheBloke/Llama-2-70B-Chat-AWQ", DEFAULT_MODEL_FILENAME),
    },
    use_hf_tokenizer=True
))


MODEL_CONFIG_REGISTRY.register(name="llama-3-8b")(HFModelConfig(
    model_name="llama-3-8b",
    chat_template_name="fallback",
    hf_base_model_repo="meta-llama/Meta-Llama-3-8B",
    quantized_model_repos={
        QuantizationMethod.GGUF: ("QuantFactory/Meta-Llama-3-8B-GGUF", f"Meta-Llama-3-8B.{GGUF_FILE_EXTENSION}"),
    },
    use_hf_tokenizer=True
))


MODEL_CONFIG_REGISTRY.register(name="llama-3-8b-instruct")(HFModelConfig(
    model_name="llama-3-8b-instruct",
    chat_template_name="llama-3-instruct",
    hf_base_model_repo="meta-llama/Meta-Llama-3-8B-Instruct",
    quantized_model_repos={
        QuantizationMethod.GGUF: ("QuantFactory/Meta-Llama-3-8B-Instruct-GGUF", f"Meta-Llama-3-8B-Instruct.{GGUF_FILE_EXTENSION}"),
    },
    use_hf_tokenizer=True
))


MODEL_CONFIG_REGISTRY.register(name="gemma-7b-it")(HFModelConfig(
    model_name="gemma-7b-it",
    chat_template_name="gemma",
    hf_base_model_repo="google/gemma-7b-it",
    quantized_model_repos={
        QuantizationMethod.GGUF: ("mlabonne/gemma-7b-it-GGUF", f"gemma-7b-it.{GGUF_FILE_EXTENSION}"),
    },
    use_hf_tokenizer=True
))
