import json
import signal
import threading
import time
from pathlib import Path
from queue import Queue
from typing import List

import psutil
from requests import Session

import subprocess


def kill_all_old_servers():
    print("Terminating any old server processes...")
    program_name = str(server_binary_path)

    """ Kills all processes that contain the program_name in their executable path. """
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            # Check if process name or the executable matches the program name
            if program_name in proc.info['name'] or (proc.info['exe'] and program_name in proc.info['exe']):
                print(f"Killing process '{proc.info['name']}' with PID {proc.info['pid']}")
                proc.send_signal(signal.SIGTERM)  # or proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass  # Process has been killed or can't be accessed


def create_completion(prompt: str, slot_id):
    request = {
        "prompt": prompt,
        "id_slot": slot_id,  # ensure that a thread only uses its own server slot
        "n_predict": 512,
        "n_probs": 1,
        "temperature": 0,
        "samplers": ["temperature"],
        "seed": 1234,
        "repeat_last_n": 0,
        "min_p": 0.0,
        "top_p": 1.0,
        "top_k": 100,
        "repeat_penalty": 1.0,
        "mirostat_eta": 0.0,
        "mirostat_tau": 0.0,
        "cache_prompt": False
    }

    raw_completion_response = session.post(url=completion_url, headers=headers, json=request).json()

    return raw_completion_response["content"]


def run_subset(thread_id: int, prompts: List[str], output_queue: Queue):
    for prompt in prompts:
        response = create_completion(prompt, thread_id)
        output_queue.put(response)


def run_all(prompts: List[str]):
    threads = []
    output_queue = Queue()

    def distribute_chunks(data, num_threads):
        n = len(data)
        chunk_size = n // num_threads
        remainder = n % num_threads

        chunks = []
        start = 0

        for thread_id in range(num_threads):
            end = start + chunk_size + (1 if thread_id < remainder else 0)
            chunks.append(data[start:end])
            start = end

        return chunks

    chunks = distribute_chunks(data=prompts, num_threads=n_parallel)

    for i in range(n_parallel):
        thread = threading.Thread(target=run_subset, args=(i, chunks[i], output_queue))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    all_results: List[str] = []
    while not output_queue.empty():
        all_results.append(output_queue.get())

    return all_results


if __name__ == '__main__':
    prompts = [
        "Once upon a time..."
    ]

    n_parallel = 2

    server_binary_path = Path("../../../llama.cpp/server")
    model_path = Path("../../../models/llama-2-7b-chat.Q4_K_M.gguf")

    completion_url = "http://localhost:8080/completion"
    headers = {'content-type': 'application/json'}

    session: Session = Session()

    kill_all_old_servers()

    # spawn a new server
    server_process_arguments = [
        str(server_binary_path),
        "-m", str(model_path),
        "-b", "1024",
        "-c", "8192",
        "-ngl", "1000",
        "-np", str(n_parallel)
    ]

    process = subprocess.Popen(server_process_arguments, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(2)

    results = run_all(prompts=prompts*16)

    print(json.dumps(results, indent=2))
