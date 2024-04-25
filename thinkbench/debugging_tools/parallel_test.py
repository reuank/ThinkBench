import asyncio
import json

import aiohttp
import time


with open('../../docs/2024-04-25_llama-2-13b-chat_reasoning-prompts.json', 'r') as f:
    prompts = json.load(f)

url = "http://localhost:8080/v1/completions"
headers = {'content-type': 'application/json'}


def get_request(prompt, backend):
    if backend == "llama.cpp":
        return {
            "prompt": prompt,
            "n_predict": 2048
        }
    elif backend == "llama-cpp-python":
        return {
            "prompt": prompt,
            "max_tokens": 2048,
            "logprobs": 1,
            "temperature": 0.0
        }


async def get_single(request, session):
    try:
        async with session.post(url=url, headers=headers, json=request) as response:
            return await response.json()
    except Exception as e:
        print("Unable to get url {} due to {}.".format(url, e.__class__))


async def run_all(count, backend):
    requests = [get_request(prompt, backend) for prompt in prompts[:count]]

    async with aiohttp.ClientSession() as session:
        async_results = await asyncio.gather(
            *(get_single(request, session) for request in requests))
    print("Finalized all. Return is a list of len {} outputs.".format(len(async_results)))
    print(async_results)


if __name__ == '__main__':
    start = time.time()
    asyncio.run(run_all(10, "llama-cpp-python"))
    end = time.time()
    print(end-start)
