from typing import List, Dict

import jinja2
from jinja2 import Template, FileSystemLoader, Environment

from benchmark import NonCoTStandardBenchmark, CoTStandardBenchmark, Benchmark
from dataset import SingleDataInstance
from inference import MessageHistory
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


def test_template_on_benchmark(template_name: str, benchmark: Benchmark, add_system_message: bool = True):
    system_message = {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    }

    single_data_instance = SingleDataInstance.get_dummy()
    prompt_chains = benchmark.prompt_chains(single_data_instance=single_data_instance)

    generated_messages = convert_chain_to_messages(prompt_chains, single_data_instance)

    if add_system_message:
        generated_messages.insert(0, system_message)

    chat_template = load_template(template_name)
    prompt = chat_template.render(messages=generated_messages, bos_token="<s>", eos_token="</s>", add_generation_prompt=True)

    return prompt


def test_all_templates_with_all_benchmarks():
    template_names = [
        "llama-2-chat",
        "orca-2"
    ]

    benchmarks = [
        NonCoTStandardBenchmark(),
        CoTStandardBenchmark()
    ]

    for template_name in template_names:
        for benchmark in benchmarks:
            print("="*50)
            print(f"Testing Template {template_name} on {benchmark.__class__.__name__}:")
            print("=" * 50)
            print(test_template_on_benchmark(template_name, benchmark))

if __name__ == '__main__':
    test_all_templates_with_all_benchmarks()
