# -*- coding: utf-8 -*-

# @Time    : 2024/4/15 11:20
# @Author  : kewei

import orjson
import httpx
from loguru import logger
from typing import Optional
from app.relay.base import RelayBase
from app.schemas.relay import ChatBody, APIResponse, StreamChatCompletion, \
    ChatCompletion, APIBody, EmbeddingBody, EmbeddingResponse, EmbeddingData
from app.utils.common import strip_prefix, encode_embedding
from app.utils.crud.models import ChannelKey
from app.relay.schemas import RequestType
from app.utils.exceptions import OtherException


class BaichuanRelay(RelayBase):
    def __init__(self):
        super().__init__()
        self.is_stream = False
        self.request_message: str = ''
        self.response_text: str = ''
        self.request_type: RequestType = RequestType.chat

        self.encoding_format: Optional[str] = None

    def get_model_list(self):
        return {'Baichuan2-Turbo', 'Baichuan2-Turbo-192k', 'Baichuan-Text-Embedding'}

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

        chat_body.model = api_model
        json_data = chat_body.dict(exclude_defaults=True)

        url = api_base + '/chat/completions'

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }

        request = httpx.Request('POST', url,
                                headers=headers,
                                json=json_data)

        self.request_message = '\n'.join(d['content'] for d in chat_body.messages)
        self.is_stream = chat_body.stream
        self.source_model = chat_body.model
        self.model = json_data['model']
        self.key_type = key.key_type

        logger.info(f"BaichuanRelay.convert_request: {request}")

        return request

    def build_embedding_request(self, embedding_body: EmbeddingBody, key: ChannelKey) -> httpx.Request:
        api_key = key.api_key
        api_base = key.api_base
        api_model = key.api_model if key.api_model != '*' else embedding_body.model
        url = api_base + '/embeddings'
        headers = {
            'Authorization': f'Bearer {api_key}',
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

        logger.info(f"BaichuanRelay.convert_request embedding: {request}")

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

        stream_chunk = StreamChatCompletion.parse_raw(chunk)
        stream_chunk.model = self.source_model
        self.response_text += stream_chunk.choices[0].delta.get('content') or ''
        return stream_chunk

    def convert_chat_response(self, chunk: str) -> Optional[ChatCompletion]:

        if isinstance(chunk, str) and not chunk.strip():
            return None

        chunk = ChatCompletion.parse_raw(chunk)
        chunk.model = self.source_model
        self.response_text += chunk.choices[0].message.get('content') or ''
        return chunk

    def convert_embedding_response(self, chunk: str) -> Optional[EmbeddingResponse]:
        if isinstance(chunk, str) and not chunk.strip():
            return None

        chunk = EmbeddingResponse.parse_raw(chunk)
        chunk.model = self.source_model
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
