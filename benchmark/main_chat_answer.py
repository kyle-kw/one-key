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
import pandas as pd
import re

current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_PATH = "logs/benchmark-sync.log"
OUTPUT_LOG_PATH = f"logs/benchmark-result-{current_dt}.csv"
RESULT_OUTPUT_PATH = "output"

# 设置loguru日志等级
logger.remove()
logger.add(LOG_PATH, level="DEBUG")


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
    answer: Any = None
    other_info: Any = None


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
    correct_rate: float


@dataclass
class RequestFuncOutput:
    prompt_text: str = ""
    generated_text: str = ""
    success: bool = False
    latency: float = 0
    ttft: float = 0
    prompt_len: int = 0
    input_tokens: int = 0
    answer: str = None
    other_info: str = None
    return_success: bool = False

    def dict(self):
        return {
            "prompt_text": self.prompt_text,
            "generated_text": self.generated_text,
            "success": self.success,
            "latency": self.latency,
            "ttft": self.ttft,
            "prompt_len": self.prompt_len,
            "input_tokens": self.input_tokens,
            "answer": self.answer,
            "other_info": self.other_info,
            "return_success": self.return_success
        }


PROMPT_TEMPLATE = """\
请仔细阅读以下问题和选项，然后根据您的知识和理解，选择最合适的答案。

问题：{question}

A. {option_a}

B. {option_b}

C. {option_c}

D. {option_d}

请思考并选择你认为最符合问题答案的选项。并返回对应的选项字母，如：A、B、C、D。
**只返回对应的选项的字母！**
"""

# 测试数据清单
TASK_NAME_MAPPING = {
    "computer_network": ["Computer Network", "\u8ba1\u7b97\u673a\u7f51\u7edc", "STEM"],
    "operating_system": ["Operating System", "\u64cd\u4f5c\u7cfb\u7edf", "STEM"],
    "computer_architecture": [
        "Computer Architecture",
        "\u8ba1\u7b97\u673a\u7ec4\u6210",
        "STEM",
    ],
    "college_programming": ["College Programming", "\u5927\u5b66\u7f16\u7a0b", "STEM"],
    "college_physics": ["College Physics", "\u5927\u5b66\u7269\u7406", "STEM"],
    "college_chemistry": ["College Chemistry", "\u5927\u5b66\u5316\u5b66", "STEM"],
    "advanced_mathematics": [
        "Advanced Mathematics",
        "\u9ad8\u7b49\u6570\u5b66",
        "STEM",
    ],
    "probability_and_statistics": [
        "Probability and Statistics",
        "\u6982\u7387\u7edf\u8ba1",
        "STEM",
    ],
    "discrete_mathematics": [
        "Discrete Mathematics",
        "\u79bb\u6563\u6570\u5b66",
        "STEM",
    ],
    "electrical_engineer": [
        "Electrical Engineer",
        "\u6ce8\u518c\u7535\u6c14\u5de5\u7a0b\u5e08",
        "STEM",
    ],
    "metrology_engineer": [
        "Metrology Engineer",
        "\u6ce8\u518c\u8ba1\u91cf\u5e08",
        "STEM",
    ],
    "high_school_mathematics": [
        "High School Mathematics",
        "\u9ad8\u4e2d\u6570\u5b66",
        "STEM",
    ],
    "high_school_physics": ["High School Physics", "\u9ad8\u4e2d\u7269\u7406", "STEM"],
    "high_school_chemistry": [
        "High School Chemistry",
        "\u9ad8\u4e2d\u5316\u5b66",
        "STEM",
    ],
    "high_school_biology": ["High School Biology", "\u9ad8\u4e2d\u751f\u7269", "STEM"],
    "middle_school_mathematics": [
        "Middle School Mathematics",
        "\u521d\u4e2d\u6570\u5b66",
        "STEM",
    ],
    "middle_school_biology": [
        "Middle School Biology",
        "\u521d\u4e2d\u751f\u7269",
        "STEM",
    ],
    "middle_school_physics": [
        "Middle School Physics",
        "\u521d\u4e2d\u7269\u7406",
        "STEM",
    ],
    "middle_school_chemistry": [
        "Middle School Chemistry",
        "\u521d\u4e2d\u5316\u5b66",
        "STEM",
    ],
    "veterinary_medicine": ["Veterinary Medicine", "\u517d\u533b\u5b66", "STEM"],
    "college_economics": [
        "College Economics",
        "\u5927\u5b66\u7ecf\u6d4e\u5b66",
        "Social Science",
    ],
    "business_administration": [
        "Business Administration",
        "\u5de5\u5546\u7ba1\u7406",
        "Social Science",
    ],
    "marxism": [
        "Marxism",
        "\u9a6c\u514b\u601d\u4e3b\u4e49\u57fa\u672c\u539f\u7406",
        "Social Science",
    ],
    "mao_zedong_thought": [
        "Mao Zedong Thought",
        "\u6bdb\u6cfd\u4e1c\u601d\u60f3\u548c\u4e2d\u56fd\u7279\u8272\u793e\u4f1a\u4e3b\u4e49\u7406\u8bba\u4f53\u7cfb\u6982\u8bba",
        "Social Science",
    ],
    "education_science": ["Education Science", "\u6559\u80b2\u5b66", "Social Science"],
    "teacher_qualification": [
        "Teacher Qualification",
        "\u6559\u5e08\u8d44\u683c",
        "Social Science",
    ],
    "high_school_politics": [
        "High School Politics",
        "\u9ad8\u4e2d\u653f\u6cbb",
        "Social Science",
    ],
    "high_school_geography": [
        "High School Geography",
        "\u9ad8\u4e2d\u5730\u7406",
        "Social Science",
    ],
    "middle_school_politics": [
        "Middle School Politics",
        "\u521d\u4e2d\u653f\u6cbb",
        "Social Science",
    ],
    "middle_school_geography": [
        "Middle School Geography",
        "\u521d\u4e2d\u5730\u7406",
        "Social Science",
    ],
    "modern_chinese_history": [
        "Modern Chinese History",
        "\u8fd1\u4ee3\u53f2\u7eb2\u8981",
        "Humanities",
    ],
    "ideological_and_moral_cultivation": [
        "Ideological and Moral Cultivation",
        "\u601d\u60f3\u9053\u5fb7\u4fee\u517b\u4e0e\u6cd5\u5f8b\u57fa\u7840",
        "Humanities",
    ],
    "logic": ["Logic", "\u903b\u8f91\u5b66", "Humanities"],
    "law": ["Law", "\u6cd5\u5b66", "Humanities"],
    "chinese_language_and_literature": [
        "Chinese Language and Literature",
        "\u4e2d\u56fd\u8bed\u8a00\u6587\u5b66",
        "Humanities",
    ],
    "art_studies": ["Art Studies", "\u827a\u672f\u5b66", "Humanities"],
    "professional_tour_guide": [
        "Professional Tour Guide",
        "\u5bfc\u6e38\u8d44\u683c",
        "Humanities",
    ],
    "legal_professional": [
        "Legal Professional",
        "\u6cd5\u5f8b\u804c\u4e1a\u8d44\u683c",
        "Humanities",
    ],
    "high_school_chinese": [
        "High School Chinese",
        "\u9ad8\u4e2d\u8bed\u6587",
        "Humanities",
    ],
    "high_school_history": [
        "High School History",
        "\u9ad8\u4e2d\u5386\u53f2",
        "Humanities",
    ],
    "middle_school_history": [
        "Middle School History",
        "\u521d\u4e2d\u5386\u53f2",
        "Humanities",
    ],
    "civil_servant": ["Civil Servant", "\u516c\u52a1\u5458", "Other"],
    "sports_science": ["Sports Science", "\u4f53\u80b2\u5b66", "Other"],
    "plant_protection": ["Plant Protection", "\u690d\u7269\u4fdd\u62a4", "Other"],
    "basic_medicine": ["Basic Medicine", "\u57fa\u7840\u533b\u5b66", "Other"],
    "clinical_medicine": ["Clinical Medicine", "\u4e34\u5e8a\u533b\u5b66", "Other"],
    "urban_and_rural_planner": [
        "Urban and Rural Planner",
        "\u6ce8\u518c\u57ce\u4e61\u89c4\u5212\u5e08",
        "Other",
    ],
    "accountant": ["Accountant", "\u6ce8\u518c\u4f1a\u8ba1\u5e08", "Other"],
    "fire_engineer": [
        "Fire Engineer",
        "\u6ce8\u518c\u6d88\u9632\u5de5\u7a0b\u5e08",
        "Other",
    ],
    "environmental_impact_assessment_engineer": [
        "Environmental Impact Assessment Engineer",
        "\u73af\u5883\u5f71\u54cd\u8bc4\u4ef7\u5de5\u7a0b\u5e08",
        "Other",
    ],
    "tax_accountant": ["Tax Accountant", "\u7a0e\u52a1\u5e08", "Other"],
    "physician": ["Physician", "\u533b\u5e08\u8d44\u683c", "Other"],
}


def read_data(file_path):
    data = pd.read_csv(file_path, sep=',', header=0)
    for _, one in data.iterrows():
        yield one


def generate_prompt(file_path_base, num=100):
    count = 0
    task_lst = list(TASK_NAME_MAPPING.keys())
    random.shuffle(task_lst)
    for task_name in task_lst:
        if count >= num:
            break
        file_path = f'{file_path_base}/{task_name}_val.csv'
        for one in read_data(file_path):
            if count >= num:
                break
            one_test = one.to_dict()
            prompt = PROMPT_TEMPLATE.format(
                question=one_test['question'],
                option_a=one_test['A'],
                option_b=one_test['B'],
                option_c=one_test['C'],
                option_d=one_test['D']
            )
            answer = one_test['answer']
            count += 1
            yield prompt, answer, TASK_NAME_MAPPING[task_name][1]


def extract_answer(result):
    res = re.findall(r'[A-D]', result)
    if res:
        return res[0]
    else:
        return None


def judge_answer(result, answer):
    return extract_answer(result) == answer


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
    output.prompt_text = chat_body.messages[0]["content"]
    output.input_tokens = chat_body.input_tokens
    output.answer = chat_body.answer
    output.other_info = chat_body.other_info
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
        output.return_success = judge_answer(generated_text, output.answer)
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
        request_throughput=round(completed / dur_s, 2),
        input_throughput=round(total_input / dur_s, 2),
        output_throughput=round(total_output / dur_s, 2),
        mean_ttft_ms=round(np.mean(ttfts) * 1000, 2),
        median_ttft_ms=round(np.median(ttfts) * 1000, 2),
        p99_ttft_ms=round(np.percentile(ttfts, 99) * 1000, 2),
        mean_tpot_ms=round(np.mean(per_token_latencies) * 1000, 2),
        median_tpot_ms=round(np.median(per_token_latencies) * 1000, 2),
        p99_tpot_ms=round(np.percentile(per_token_latencies, 99) * 1000, 2),
        correct_rate=round(sum([1 for output in outputs if output.return_success]) / len(outputs), 4)
    )

    return metrics


def get_request(
        input_requests: List[Tuple[str, str, str]],
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


def get_thread_request(input_requests: List[Tuple[str, str, str]],
                       request_rate: float,
                       model: str = 'gpt-3.5-turbo',
                       stream: bool = True):
    for request in get_request(input_requests, request_rate):
        prompt, answer, other_info = request
        chat_body = ChatBody(
            model=model,
            messages=[
                {'role': 'user', 'content': prompt},
            ],
            max_tokens=1000,
            stream=stream,
            input_tokens=get_token(prompt),
            answer=answer,
            other_info=other_info
        )
        yield chat_body


def save_logs_csv(logs: List[dict]):
    pd.DataFrame(logs).to_csv(OUTPUT_LOG_PATH, index=False)


def benchmark(
        input_requests: List[Tuple[str, str, str]],
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

    save_logs_csv([output.dict() for output in outputs])
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
    print(f"Correct Rate: {metrics.correct_rate * 100:.2f}%")

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
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "correct_rate": metrics.correct_rate
    }
    return result


def sample_requests(
        dataset_path: str,
        num_requests: int,
) -> List[Tuple[str, str, str]]:
    sampled_requests = [d for d in generate_prompt(dataset_path, num_requests)]

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
        # current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
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
            f"{RESULT_OUTPUT_PATH}/{parallel}-{model_id}-{current_dt}.json"
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

