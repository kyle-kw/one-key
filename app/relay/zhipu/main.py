# -*- coding: utf-8 -*-

# @Time    : 2024/4/12 10:21
# @Author  : kewei
# https://open.bigmodel.cn/dev/api#glm-4

import jwt
import time
import httpx
import orjson
from loguru import logger
from typing import Optional
from cachetools import cached, TTLCache
from app.relay.base import RelayBase
from app.schemas.relay import ChatBody, APIResponse, StreamChatCompletion, \
    ChatCompletion, ChatChoice, StreamChatChoice, APIBody, EmbeddingBody, EmbeddingResponse
from app.utils.common import strip_prefix, encode_embedding
from app.utils.crud.models import ChannelKey
from app.relay.schemas import RequestType
from app.utils.exceptions import OtherException


@cached(cache=TTLCache(maxsize=10, ttl=30))
def generate_token(apikey: str):
    try:
        api_key, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid api_key", e)

    payload = {
        "api_key": api_key,
        "exp": int(round(time.time() * 1000)) + 60 * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


class ZhipuRelay(RelayBase):
    def __init__(self):
        super().__init__()
        self.is_stream = False

        self.request_type: RequestType = RequestType.chat
        self.encoding_format: Optional[str] = None

    def get_model_list(self):
        return {'glm-4', 'glm-4v', 'glm-3-turbo', 'Embedding-2'}

    def build_chat_request(self, chat_body: ChatBody, key: ChannelKey) -> httpx.Request:
        """
        格式化请求，构造httpx.Request
        实例化self.request_message, self.send_request

        :param chat_body:
        :param key:
        :return:
        """
        api_key = key.api_key
        api_base = key.api_base
        api_model = key.api_model if key.api_model != '*' else chat_body.model
        self.model = api_model
        url = api_base + '/chat/completions'

        token = generate_token(api_key)
        json_data = {
            'model': api_model,
            'messages': chat_body.messages,
        }
        if chat_body.user:
            json_data['request_id'] = chat_body.user

        if chat_body.stream:
            json_data['stream'] = chat_body.stream

        if chat_body.top_p:
            json_data['top_p'] = chat_body.top_p

        if chat_body.max_tokens:
            json_data['max_tokens'] = chat_body.max_tokens

        if chat_body.stop:
            json_data['stop'] = chat_body.stop

        headers = {
            'Content-Type': 'application/json',
            'Authorization': token,
        }

        request = httpx.Request('POST', url,
                                headers=headers,
                                json=json_data)

        self.request_message = '\n'.join(d['content'] for d in chat_body.messages)
        self.is_stream = chat_body.stream
        self.source_model = chat_body.model

        logger.info(f"OpenaiRelay.convert_request: {request}")

        return request

    def build_embedding_request(self, embedding_body: EmbeddingBody, key: ChannelKey) -> httpx.Request:
        api_key = key.api_key
        api_base = key.api_base
        api_model = key.api_model if key.api_model != '*' else embedding_body.model
        url = api_base + '/embeddings'

        if isinstance(embedding_body.input, list):
            raise OtherException("ZhipuRelay embedding input must be string")

        token = generate_token(api_key)

        headers = {
            'Authorization': token,
            'Content-Type': 'application/json',
        }

        json_data = {
            "model": api_model,
            "input": embedding_body.input
        }

        request = httpx.Request('POST', url,
                                headers=headers,
                                json=json_data)

        self.request_message = (
            '\n'.join(embedding_body.input)
            if isinstance(embedding_body.input, list)
            else embedding_body.input
        )
        self.is_stream = False
        self.model = api_model
        self.key_type = key.key_type
        self.encoding_format = embedding_body.encoding_format
        self.source_model = embedding_body.model

        logger.info(f"ZhipuRelay.convert_request embedding: {request}")

        return request

    def build_request(self, api_body: APIBody, key: ChannelKey) -> httpx.Request:
        if isinstance(api_body, ChatBody):
            self.request_type = RequestType.chat
            return self.build_chat_request(api_body, key)

        elif isinstance(api_body, EmbeddingBody):
            self.request_type = RequestType.embeddings
            return self.build_embedding_request(api_body, key)

        raise NotImplementedError

    def convert_chat_stream_response(self, chunk: str) -> Optional[StreamChatCompletion]:
        if not chunk.strip() or 'data: [DONE]' == chunk:
            return None

        chunk = strip_prefix(chunk, 'data:')
        chunk = orjson.loads(chunk)
        stream_chunk = StreamChatCompletion(
            id=chunk['id'],
            object='',
            model=self.source_model,
            choices=[StreamChatChoice(
                index=0,
                delta=chunk['choices'][0]['delta'],
            )],
            created=chunk['created'],
        )
        self.response_text += stream_chunk.choices[0].delta.get('content') or ''
        return stream_chunk

    def convert_chat_response(self, chunk: str) -> Optional[ChatCompletion]:

        if isinstance(chunk, str) and not chunk.strip():
            return None

        chunk = orjson.loads(chunk)
        chunk = ChatCompletion(
            id=chunk['id'],
            object=chunk['request_id'],
            model=self.source_model,
            choices=[ChatChoice(
                index=0,
                message=chunk['choices'][0]['message'],
                finish_reason=chunk['choices'][0]['finish_reason'],
            )],
            created=chunk['created'],
            usage=chunk['usage'],
        )
        self.response_text += chunk.choices[0].message.get('content') or ''
        return chunk

    def convert_embedding_response(self, chunk: str) -> Optional[EmbeddingResponse]:
        if isinstance(chunk, str) and not chunk.strip():
            return None

        chunk = EmbeddingResponse.parse_raw(chunk)

        if self.encoding_format == 'base64':
            for d in chunk.data:
                d.embedding = encode_embedding(d.embedding)

        if chunk.usage:
            self.request_token = chunk.usage.get('prompt_tokens')
            self.response_token = 0

        return chunk

    def convert_response(self, chunk: str) -> Optional[APIResponse]:
        if self.request_type == RequestType.chat:
            if self.is_stream:
                return self.convert_chat_stream_response(chunk)
            else:
                return self.convert_chat_response(chunk)

        elif self.request_type == RequestType.embeddings:
            return self.convert_embedding_response(chunk)
