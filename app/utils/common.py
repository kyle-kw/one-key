# -*- coding: utf-8 -*-

# @Time    : 2024/4/11 15:36
# @Author  : kewei

import numpy as np
import base64
import tiktoken
import random
from string import ascii_letters, digits
from typing import List, Union
from app.utils.logger import logger
from app.config import env_settings
from app.utils.limiter import add_keys_token, limit_keys_token
from app.utils.exceptions import TokenLimitException


def strip_prefix(text: str, prefix: str) -> str:
    """
    去除字符串前缀

    :param text:  待处理字符串
    :param prefix:  前缀
    :return:  去除前缀后的字符串
    """
    if not text:
        return text

    if text.startswith(prefix):
        return text[len(prefix):]

    return text


def count_text_tokens(text, model_name="gpt-3.5-turbo"):
    """
    计算一段文本的使用的token数
    :param text:
    :param model_name:
    :return:
    """
    if not text:
        return 0

    enc = tiktoken.encoding_for_model(model_name)

    tokens = len(enc.encode(text))
    return tokens


async def auth_add_key_token(auth_key: str, tokens: int):
    if not env_settings.limit_token:
        return True

    try:
        await add_keys_token(auth_key, tokens)
    except Exception as e:
        logger.error(e)


async def auth_limit_key_token(auth_key: str, limit_token: int):
    """
    记录使用的token数
    :param auth_key:  用户key
    :param limit_token:  使用的token数
    :return:
    """
    if not env_settings.limit_token:
        return True

    limit_status = 1
    try:
        # 限流
        limit_status = await limit_keys_token(auth_key, limit_token)
    except Exception as e:
        logger.error(e)

    if limit_status == 0:
        raise TokenLimitException()

    return True


def decode_embedding(embedding_str):
    """
    解码embedding结果

    :param embedding_str: base64编码的embedding结果
    :return:  解码后的embedding结果
    """
    embedding_list = np.frombuffer(
        base64.b64decode(embedding_str), dtype="float32"
    ).tolist()

    return embedding_list


def encode_embedding(embedding_list):
    """
    编码embedding结果

    :param embedding_list:  embedding结果
    :return:  base64编码的embedding结果
    """
    encoded_str = base64.b64encode(
        np.array(embedding_list, dtype="float32").tobytes()
    ).decode("utf-8")

    return encoded_str


def get_random_string(length: int) -> str:
    """
    生成随机字符串
    :param length: 字符串长度
    :return:
    """
    return ''.join(random.choices(ascii_letters + digits, k=length))


def generate_fd_key():
    """
    随机生成auth key
    :return:
    """
    return "fd-" + get_random_string(47)


def decode_token_char(token_char: List[int], model_name: str = "cl100k_base"):
    """
    解码token字符

    :param token_char:
    :param model_name:
    :return:
    """
    enc = tiktoken.get_encoding(model_name)
    return enc.decode(token_char)


def convert_input_data(input_data: Union[List[Union[List[int], int, str]], str]) -> List[str]:
    """
    转换输入数据为统一的格式

    :param input_data:
    :return:
    """
    if not input_data:
        return input_data

    if isinstance(input_data, str):
        return [input_data]
    elif isinstance(input_data, list):
        if isinstance(input_data[0], str):
            return input_data
        elif isinstance(input_data[0], int):
            return [decode_token_char(input_data)]
        elif isinstance(input_data[0], list):
            return [decode_token_char(d) for d in input_data]
    raise Exception("input 类型错误。")


def convert_embedding_data(data: Union[List[int], str], encoding_format: str) -> Union[List[int], str]:
    """
    转换embedding数据格式
    :param data:
    :param encoding_format:
    :return:
    """
    if not data:
        return data

    if not encoding_format:
        encoding_format = 'float'

    if encoding_format == 'float' and isinstance(data, str):
        return decode_embedding(data)
    elif encoding_format == 'base64' and isinstance(data, list):
        return encode_embedding(data)
    return data

