# -*- coding: utf-8 -*-

# @Time    : 2024/4/10 16:32
# @Author  : kewei

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Generator, List, Optional, Mapping, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import httpx
import numpy as np
import tiktoken
from httpx import TimeoutException, RequestError
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

# 设置loguru日志等级
logger.remove()
logger.add("logs/benchmark-sync.log", level="DEBUG")


class ChatBody(BaseModel):
    model: str = Field(...)
    messages: List[Mapping[str, str]] = Field(...)
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None

    input_tokens: Any = None


class ChannelKey(BaseModel):
    api_key: str
    api_base: str = ''
    api_model: str
    api_config: Optional[dict]


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0
    ttft: float = 0
    prompt_len: int = 0
    input_tokens: int = 0


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def generate_request(chat_body: ChatBody) -> httpx.Request:
    api_key = api_key_model.api_key
    api_base = api_key_model.api_base

    json_data = chat_body.model_dump(exclude_defaults=True)

    url = api_base + '/chat/completions'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'api-key': api_key
    }

    request = httpx.Request('POST', url,
                            headers=headers,
                            json=json_data)

    return request


def request_openai_chat_completions(
        chat_body: ChatBody,
        pbar: Optional[tqdm] = None,
):
    request = generate_request(chat_body)

    generated_text = ""
    ttft = 0
    st = time.perf_counter()
    output = RequestFuncOutput()
    output.input_tokens = chat_body.input_tokens
    latency = 0
    try:
        for chunk in _stream_request(request):
            if ttft == 0:
                ttft = time.perf_counter() - st
                output.ttft = ttft

            chunk = chunk.strip()
            if not chunk:
                continue

            chunk = remove_prefix(chunk, "data: ")
            if chunk == "[DONE]":
                latency = time.perf_counter() - st
            else:
                body = json.loads(chunk)
                if "content" in body["choices"][0]["delta"]:
                    generated_text += body["choices"][0]["delta"]["content"]

        output.generated_text = generated_text
        output.success = True
        output.latency = latency

    except (RequestError, TimeoutException, Exception):
        output.success = False

    if pbar:
        pbar.update(1)
    return output


def _stream_request(request: httpx.Request) -> Iterator[str]:
    with httpx.Client() as client:
        try:
            response = client.send(request, stream=True)
        except httpx.TimeoutException as err:
            logger.debug("Encountered httpx.TimeoutException", exc_info=True)

            logger.debug("Raising timeout error")
            raise TimeoutException('request timeout') from err
        except Exception as err:
            logger.debug("Encountered Exception", exc_info=True)

            logger.debug("Raising connection error")
            raise RequestError('network error') from err

        logger.debug(f'HTTP Request: {request.method} {request.url} "{response.status_code} {response.reason_phrase}"')

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:  # thrown on 4xx and 5xx status code
            logger.debug("Encountered httpx.HTTPStatusError", exc_info=True)
            logger.debug("Re-raising status error")

            error_message = f"HTTP status error: {response.status_code} {response.reason_phrase}"
            raise RequestError(error_message) from None

        for chunk in response.iter_lines():
            yield chunk

        response.close()


def get_token(text: str, model_name: str = 'gpt-3.5-turbo'):
    encoding = tiktoken.encoding_for_model(model_name)

    return len(encoding.encode(text))


def calculate_metrics(
        outputs: List[RequestFuncOutput],
        dur_s: float,
) -> BenchmarkMetrics:
    total_output = 0
    total_input = 0
    completed = 0
    per_token_latencies = []
    ttfts = []
    for i in range(len(outputs)):
        if not outputs[i].generated_text:
            outputs[i].success = False
            continue

        if outputs[i].success:
            output_len = get_token(outputs[i].generated_text)
            total_output += output_len
            total_input += outputs[i].input_tokens
            per_token_latencies.append(outputs[i].latency / output_len)
            ttfts.append(outputs[i].ttft)
            completed += 1

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=total_output,
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=total_output / dur_s,
        mean_ttft_ms=np.mean(ttfts) * 1000,
        median_ttft_ms=np.median(ttfts) * 1000,
        p99_ttft_ms=np.percentile(ttfts, 99) * 1000,
        mean_tpot_ms=np.mean(per_token_latencies) * 1000,
        median_tpot_ms=np.median(per_token_latencies) * 1000,
        p99_tpot_ms=np.percentile(per_token_latencies, 99) * 1000,
    )

    return metrics


def get_request(
        input_requests: List[Tuple[str, int, int]],
        request_rate: float,
) -> Generator[Tuple[str, int, int], None, None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        time.sleep(interval)


def get_thread_request(input_requests: List[Tuple[str, int, int]],
                       request_rate: float,
                       model: str = 'gpt-3.5-turbo',
                       stream: bool = True):
    for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        chat_body = ChatBody(
            model=model,
            messages=[
                {'role': 'user', 'content': prompt},
            ],
            max_tokens=output_len,
            stream=stream,
            input_tokens=prompt_len
        )
        yield chat_body


def benchmark(
        input_requests: List[Tuple[str, int, int]],
        request_rate: float,
        model: str = 'gpt-3.5-turbo',
        stream: bool = True,
        parallel: int = 10,
):
    pbar = tqdm(total=len(input_requests))
    benchmark_start_time = time.perf_counter()

    pool = ThreadPoolExecutor(parallel)

    outputs = []
    for r in pool.map(partial(request_openai_chat_completions, pbar=pbar),
                      get_thread_request(input_requests, request_rate, model, stream)):
        outputs.append(r)

    pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
    )

    print(f"Successful requests: {metrics.completed}")
    print(f"Benchmark duration: {benchmark_duration:2f} s")
    print(f"Total input tokens: {metrics.total_input}")
    print(f"Total generated tokens: {metrics.total_output}")
    print(f"Request throughput: {metrics.request_throughput:.2f} requests/s")
    print(f"Input token throughput: {metrics.input_throughput:.2f} tokens/s")
    print(f"Output token throughput: {metrics.output_throughput:.2f} tokens/s")
    print(f"Mean TTFT: {metrics.mean_ttft_ms:.2f} ms")
    print(f"Median TTFT: {metrics.median_ttft_ms:.2f} ms")
    print(f"P99 TTFT: {metrics.p99_ttft_ms:.2f} ms")
    print(f"Mean TPOT: {metrics.mean_tpot_ms:.2f} ms")
    print(f"Median TPOT: {metrics.median_tpot_ms:.2f} ms")
    print(f"P99 TPOT: {metrics.p99_tpot_ms:.2f} ms")

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_inthroughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms
    }
    return result


def sample_requests(
        dataset_path: str,
        num_requests: int,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # some of these will be filtered out, so sample more than we need
    sampled_indices = random.sample(range(len(dataset)),
                                    int(num_requests * 1.2))
    dataset = [dataset[i] for i in sampled_indices]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    completions = [completion for _, completion in dataset]
    tokenized_dataset = []
    for i in range(len(dataset)):
        tokenized_dataset.append((prompts[i], get_token(completions[i])))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, output_len in tokenized_dataset:
        prompt_len = get_token(prompt)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_id = args.model
    stream = args.stream
    parallel = args.parallel

    input_requests = sample_requests(args.dataset, args.num_prompts)

    benchmark_result = benchmark(
        input_requests=input_requests,
        request_rate=args.request_rate,
        model=model_id,
        stream=stream,
        parallel=parallel,
    )

    # Save config and results to json
    if args.save_result:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt

        result_json["version"] = args.version
        result_json["model_id"] = model_id
        result_json["num_prompts"] = args.num_prompts

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        file_name = (
            f"output/chat-{parallel}-{model_id}-{current_dt}.json"
        )
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")

    parser.add_argument(
        "--version",
        type=str,
        default="N/A",
        help="Version of the serving backend/engine.",
    )
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
             "then all the requests are sent at time 0. "
             "Otherwise, we use Poisson process to synthesize "
             "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )

    parser.add_argument("--api-key",
                        type=str,
                        required=True,
                        help="API key for the model.")
    parser.add_argument("--api-base",
                        type=str,
                        required=True,
                        help="API base for the model.")

    parser.add_argument("--stream",
                        type=bool,
                        default=True,
                        help="Whether to stream the output.")

    parser.add_argument("--parallel",
                        type=int,
                        default=10,
                        help="Number of parallel requests.")

    args = parser.parse_args()

    api_key_model = ChannelKey(
        api_key=args.api_key,
        api_base=args.api_base,
        api_model=args.model,
        api_config=None
    )

    main(args)
