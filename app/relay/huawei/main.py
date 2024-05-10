# -*- coding: utf-8 -*-

# @Time    : 2024/4/16 15:50
# @Author  : kewei

import time
import httpx
import orjson
from loguru import logger
from typing import Optional
from cachetools import cached, TTLCache
from app.relay.base import RelayBase
from app.schemas.relay import ChatBody, APIResponse, StreamChatCompletion, \
    ChatCompletion, APIBody, ChatChoice, StreamChatChoice
from app.utils.common import strip_prefix
from app.utils.crud.models import ChannelKey
from app.utils.exceptions import OtherException


@cached(cache=TTLCache(maxsize=10, ttl=30))
def get_api_token(ak, sk):
    auth_data = {
        "auth": {
            "identity": {
                "methods": [
                    "hw_ak_sk"
                ],
                "hw_ak_sk": {
                    "access": {
                        "key": ak
                    },
                    "secret": {
                        "key": sk
                    }
                }
            },
            "scope": {
                "project": {
                    "name": "cn-southwest-2"
                }
            }
        }
    }
    AUTH_URL = r'https://iam.myhuaweicloud.com/v3/auth/tokens'
    resp = httpx.post(AUTH_URL, json=auth_data, headers={"Content-Type": "application/json"})
    token = resp.headers.get('X-Subject-Token')
    if token:
        return token
    else:
        raise Exception(f"huawei Auth failed: {resp.status_code}")


class HuaweiRelay(RelayBase):
    def __init__(self):
        super().__init__()
        self.is_stream = False

    def get_model_list(self):
        return {}

    def build_request(self, api_body: APIBody, key: ChannelKey) -> httpx.Request:
        """
        格式化请求，构造httpx.Request
        实例化self.request_message, self.send_request
        """
        if not isinstance(api_body, ChatBody):
            raise ValueError("APIBody should be ChatBody")

        chat_body: ChatBody = api_body

        api_key = key.api_key
        api_base = key.api_base
        api_model = key.api_model if key.api_model != '*' else chat_body.model

        ak, sk = api_key.split('/')
        token = get_api_token(ak, sk)

        json_data = {
            'messages': chat_body.messages,
        }

        if chat_body.max_tokens:
            json_data['max_tokens'] = min(max(chat_body.max_tokens, 100), 2000)

        if chat_body.stream:
            json_data['stream'] = chat_body.stream

        if chat_body.temperature:
            json_data['temperature'] = min(max(chat_body.temperature, 0.1), 1)

        if chat_body.top_p:
            json_data['top_p'] = chat_body.top_p

        if chat_body.user:
            json_data['user'] = chat_body.user

        if chat_body.presence_penalty:
            json_data['presence_penalty'] = chat_body.presence_penalty

        if chat_body.n:
            json_data['n'] = chat_body.n

        url = api_base + '/chat/completions'

        headers = {
            'content-type': 'application/json',
            'X-Auth-Token': token
        }

        request = httpx.Request('POST', url,
                                headers=headers,
                                json=json_data)

        self.request_message = '\n'.join(d['content'] for d in chat_body.messages)
        self.is_stream = chat_body.stream
        self.model = api_model
        self.key_type = key.key_type
        self.source_model = chat_body.model

        logger.info(f"HuaweiRelay.convert_request: {request}")

        return request

    def convert_chat_stream_response(self, chunk: str) -> Optional[StreamChatCompletion]:
        if not chunk.strip() or chunk == 'data:[DONE]':
            return None

        if chunk.startswith('event'):
            chunk = strip_prefix(chunk, 'event:')
            chunk = orjson.loads(chunk)
            self.response_token = chunk['token_number']
            return None

        chunk = strip_prefix(chunk, 'data:')
        chunk = orjson.loads(chunk)
        if 'error_code' in chunk:
            raise OtherException(str(chunk['error_msg']))

        stream_chunk = StreamChatCompletion(
            id=chunk['id'],
            object='',
            model=self.source_model,
            choices=[StreamChatChoice(
                index=0,
                delta={'role': 'assistant', 'content': chunk['choices'][0]['message']['content']}
            )],
            created=chunk['created'],
        )
        self.response_text += stream_chunk.choices[0].delta.get('content') or ''
        return stream_chunk

    def convert_chat_response(self, chunk: str) -> Optional[ChatCompletion]:

        if isinstance(chunk, str) and not chunk.strip():
            return None

        chunk = orjson.loads(chunk)

        if 'error_code' in chunk:
            raise OtherException(chunk['error_msg'])

        chunk = ChatCompletion(
            id=chunk['id'],
            object='',
            model=self.source_model,
            choices=[ChatChoice(
                index=0,
                message={'role': 'assistant', 'content': chunk['choices'][0]['message']['content']},
            )],
            created=chunk['created'],
            usage=chunk['usage'],
        )
        self.response_text += chunk.choices[0].message.get('content') or ''
        return chunk

    def convert_response(self, chunk: str) -> Optional[APIResponse]:
        if self.is_stream:
            return self.convert_chat_stream_response(chunk)
        else:
            return self.convert_chat_response(chunk)
