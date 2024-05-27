import asyncio
import json
import math
from typing import List, Dict

import requests

import aiohttp
import time

from inference.completion import Choice, Logprobs, Usage, CompletionResult
from storage import TotalResultEncoder

with open('../../docs/2024-04-25_llama-2-13b-chat_reasoning-prompts.json', 'r') as f:
    prompts = json.load(f)

completion_url = "http://localhost:8080/completion"
health_url = "http://localhost:8080/health"
headers = {'content-type': 'application/json'}


def get_request(prompt, backend):
    if backend == "llama.cpp":
        return {
            "prompt": prompt,
            "n_predict": 1024,
            "n_probs": 3,
            "temperature": 1.0,
            "samplers": ["temperature"],
            "repeat_last_n": 0,
            "min_p": 0.0,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "mirostat_eta": 0.0,
            "mirostat_tau": 0.0,
            #"grammar": "root ::= (\"A\"|\"B\"|\"C\"|\"D\")"
        }
    elif backend == "llama-cpp-python":
        return {
            "prompt": prompt,
            "max_tokens": 2048,
            "logprobs": 1,
            "temperature": 0.0
        }


########


async def async_get_single(request, session):
    try:
        async with session.post(url=completion_url, headers=headers, json=request) as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        print(f"Unable to get url {completion_url} due to {e.__class__}: {e}.")
        return None


async def async_run_all(count, backend):
    requests = [get_request(prompt, backend) for prompt in prompts[:count]]

    async with aiohttp.ClientSession() as session:
        async_results = await asyncio.gather(*(async_get_single(request, session) for request in requests))
    print("Finalized all. Return is a list of len {} outputs.".format(len(async_results)))
    print(json.dumps(async_results, indent=2))


########

def __convert_completion_response(prompt: str, completion_response: Dict) -> CompletionResult:
    finish_reason = ""
    if completion_response["stopped_eos"]:
        finish_reason = "eos"
    elif completion_response["stopped_word"]:
        finish_reason = "stop"
    elif completion_response["limit"]:
        finish_reason = "length"

    top_logprobs: List[Dict[str, float]] = []
    for completion_probability in completion_response["completion_probabilities"]:
        top_logprobs.append(
            {item["tok_str"]:math.log(item["prob"]) for item in completion_probability["probs"]}
        )

    choices = [Choice(
        text=completion_response["content"],
        index=0,
        logprobs=Logprobs(
            tokens=[],
            text_offset=[],
            token_logprobs=[],
            top_logprobs=top_logprobs
        ),
        finish_reason=finish_reason
    )]

    timings = completion_response["timings"]
    usage = Usage(
        prompt_tokens=timings["prompt_n"],
        prompt_tokens_per_second=timings["prompt_per_second"],
        prompt_ms=timings["prompt_ms"],
        completion_tokens=timings["predicted_n"],
        completion_tokens_per_second=timings["predicted_per_second"],
        completion_ms=timings["predicted_ms"],
        total_tokens=timings["prompt_n"]+timings["predicted_n"]
    )

    return CompletionResult(
        prompt=prompt,
        id=completion_response['id_slot'],
        object="text_completion",
        created=int(time.time()),
        model=completion_response['model'],
        choices=choices,
        usage=usage
    )


def get_single(request, session):
    response = session.post(url=completion_url, headers=headers, json=request).json()
    return __convert_completion_response("test", response)


def run_all(count, backend):
    all_requests = [get_request(prompt, backend) for prompt in prompts[:count]]

    with requests.Session() as session:
        results = [get_single(request, session) for request in all_requests]


    print("Finalized all. Return is a list of len {} outputs.".format(len(results)))
    print(json.dumps(results, indent=2, cls=TotalResultEncoder))


########


if __name__ == '__main__':
    import subprocess

    output_file_path = 'output.log'

    with open(output_file_path, 'w') as output_file:
        process = subprocess.Popen(
            ["../../../llama.cpp/server", "-m", "../../../models/gemma-7b-it.Q4_K_M.gguf", "-c", "4096", "-ngl", "1000",
             "-np", "1", "-cb", "--log-disable"], stdout=output_file, stderr=subprocess.STDOUT)

    time.sleep(3)
    start = time.time()
    run_all(1, "llama.cpp")
    #asyncio.run(async_run_all(1, "llama.cpp"))

    end = time.time()
    process.terminate()
    print(end-start)
