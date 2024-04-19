from typing import List, Dict

from transformers import AutoTokenizer
from jinja2 import Template, FileSystemLoader, Environment

from benchmark import NonCoTStandardBenchmark, CoTStandardBenchmark, Benchmark
from dataset import SingleDataInstance
from inference import MessageHistory
from model import ModelConfig, HFModelConfig
from prompt import PromptTemplateStep, PromptCompletionStep, PromptTextStep


def load_template(template_name: str) -> Template:
    template_loader = FileSystemLoader(searchpath="./chat_templates")
    template_env = Environment(loader=template_loader, trim_blocks=True, lstrip_blocks=True)
    template_file = f"{template_name}.jinja"

    return template_env.get_template(template_file)


def convert_chain_to_messages(prompt_chains, single_data_instance) -> List[Dict]:
    selected_prompt_chain = prompt_chains[0]
    message_history = MessageHistory()

    for prompt_step in selected_prompt_chain.steps:
        if isinstance(prompt_step, PromptTemplateStep):
            prompt_step: PromptTemplateStep
            message = prompt_step.template.render(single_data_instance=single_data_instance)
            message_history.append_to_last_user_message(message)
        elif isinstance(prompt_step, PromptCompletionStep):
            prompt_step: PromptCompletionStep
            completion = prompt_step.prefix + f"[COMPLETION with {prompt_step.decoder.__class__.__name__}]"
            message_history.add_assistant_message(completion)
        elif isinstance(prompt_step, PromptTextStep):
            prompt_step: PromptTextStep
            message_history.append_to_last_user_message(prompt_step.text)
        else:
            raise ValueError(f"Prompt step type {type(prompt_step)} not implemented.")

    return message_history.messages


def test_template_on_benchmark(model_config: HFModelConfig, benchmark: Benchmark, add_system_message: bool = True):
    system_message = {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    }

    single_data_instance = SingleDataInstance.get_dummy()
    prompt_chains = benchmark.prompt_chains(single_data_instance=single_data_instance)

    generated_messages = convert_chain_to_messages(prompt_chains, single_data_instance)

    if add_system_message:
        generated_messages.insert(0, system_message)

    chat_template = model_config.chat_template
    my_prompt = chat_template.render(messages=generated_messages, bos_token="<s>", eos_token="</s>", add_generation_prompt=True)

    hf_tokenizer = AutoTokenizer.from_pretrained(model_config.hf_tokenizer, use_fast=False)
    hf_tokenized_chat = hf_tokenizer.apply_chat_template(generated_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    return my_prompt, hf_tokenizer.decode(hf_tokenized_chat[0])


def get_hf_tokenizer_reference(hf_model_path: str, messages: []):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    return tokenizer.decode(tokenized_chat[0])


def test_all_templates_with_many_benchmarks(benchmarks):
    for model_name in ModelConfig.get_all_names():
        model_config: HFModelConfig = ModelConfig.get_by_name(model_name)

        for benchmark in benchmarks:
            print("="*50)
            print("="*50)
            print(f"Testing Template {model_config.chat_template_name} on {benchmark.__class__.__name__}:")
            print("=" * 50)
            my_output, hf_reference = test_template_on_benchmark(model_config, benchmark)
            print(my_output)
            print("=" * 50)
            print(hf_reference)


def compare_my_and_hf(model_name: str):
    print("=" * 50)
    print("My Output:")
    print("=" * 50)
    my_output, hf_reference = test_template_on_benchmark(ModelConfig.get_by_name(model_name),
                                                         CoTStandardBenchmark())
    print(my_output)
    print("=" * 50)
    print("HF Reference:")
    print("=" * 50)
    print(hf_reference)
    print("=" * 50)
    print(f"Do they match? {my_output == hf_reference}")

if __name__ == '__main__':
    benchmarks = [
        NonCoTStandardBenchmark(),
        CoTStandardBenchmark()
    ]

    # get_hf_tokenizer_reference("meta-llama/Llama-2-7b-chat-hf", )
    # test_all_templates_with_many_benchmarks(benchmarks)

    compare_my_and_hf("phi-2")
    #compare_my_and_hf("llama-2-7b-chat")