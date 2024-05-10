# -*- coding: utf-8 -*-

import httpx
import time
import orjson
from loguru import logger
from typing import Optional
from app.relay.base import RelayBase
from app.schemas.relay import ChatBody, APIResponse, StreamChatCompletion, \
    ChatCompletion, ChatChoice, StreamChatChoice, APIBody, EmbeddingBody, EmbeddingResponse, EmbeddingData
from app.utils.common import strip_prefix, encode_embedding
from app.relay.schemas import RequestType
from app.utils.exceptions import OtherException
from app.utils.crud.models import ChannelKey


class AliRelay(RelayBase):
    def __init__(self):
        super().__init__()
        self.is_stream = False
        self.index: int = 0
        self.req_id: str = ''
        self.created: int = int(time.time())
        self.request_type: RequestType = RequestType.chat

        self.encoding_format: Optional[str] = None

    def get_model_list(self):
        return {'qwen-turbo', 'qwen-plus', 'qwen-max',
                'qwen-max-longcontext', 'text-embedding-v1', 'text-embedding-v2'}

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
        messages = chat_body.messages

        url = api_base or 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'

        json_data = {
            "model": api_model,
            "parameters": {
                "result_format": "message",
            },
            "input": {
                "messages": messages
            }
        }
        if chat_body.max_tokens:
            json_data['parameters']['max_tokens'] = chat_body.max_tokens
        if chat_body.stop:
            json_data['parameters']['stop'] = chat_body.stop
        if chat_body.temperature:
            json_data['parameters']['temperature'] = chat_body.temperature
        if chat_body.top_p:
            json_data['parameters']['top_p'] = chat_body.top_p
        if chat_body.seed:
            json_data['parameters']['seed'] = chat_body.seed

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

        if chat_body.stream:
            headers['Accept'] = 'text/event-stream'

        request = httpx.Request('POST', url,
                                headers=headers,
                                json=json_data)

        self.request_message = '\n'.join(d['content'] for d in chat_body.messages)
        self.is_stream = chat_body.stream
        self.model = api_model
        self.key_type = key.key_type
        self.source_model = chat_body.model

        logger.info(f"AliRelay.convert_request chat: {request}")

        return request

    def build_embedding_request(self, embedding_body: EmbeddingBody, key: ChannelKey) -> httpx.Request:
        api_key = key.api_key
        api_base = key.api_base
        api_model = key.api_model if key.api_model != '*' else embedding_body.model
        url = api_base or 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        if isinstance(embedding_body.input, str):
            text = [embedding_body.input]
        else:
            text = embedding_body.input

        json_data = {
            "model": api_model,
            "input": {
                "texts": text
            }
        }

        request = httpx.Request('POST', url,
                                headers=headers,
                                json=json_data)
        self.request_message = '\n'.join(text)
        self.is_stream = False
        self.model = api_model
        self.key_type = key.key_type
        self.encoding_format = embedding_body.encoding_format
        self.source_model = embedding_body.model

        logger.info(f"AliRelay.convert_request embedding: {request}")

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
        if not chunk.strip() or chunk.startswith(':HTTP_STATUS') or chunk.startswith('event:'):
            return None

        if chunk.startswith('id:'):
            if not self.req_id:
                self.req_id = strip_prefix(chunk, 'id:')
            return None

        chunk = strip_prefix(chunk, 'data:')

        try:
            chunk = orjson.loads(chunk)
        except orjson.JSONDecodeError as e:
            logger.error(f'AliRelay.convert_chat_stream_response: {e}')
            raise OtherException(detail=chunk)

        if 'code' in chunk:
            raise OtherException(detail=chunk['message'])

        self.request_token = chunk['usage']['input_tokens']
        self.response_token = chunk['usage']['output_tokens']

        content = chunk['output']['choices'][0]['message']['content']

        stream_chunk = StreamChatCompletion(
            id=chunk['request_id'],
            object='',
            model=self.source_model,
            choices=[StreamChatChoice(
                index=0,
                delta={
                    'content': content[self.index:],
                    'role': 'assistant'
                },
            )],
            created=self.created,
        )
        self.index = len(content)
        self.response_text += stream_chunk.choices[0].delta.get('content') or ''

        return stream_chunk

    def convert_chat_response(self, chunk: str) -> Optional[ChatCompletion]:
        if isinstance(chunk, str) and not chunk.strip():
            return None

        chunk = orjson.loads(chunk)

        if 'code' in chunk:
            raise OtherException(detail=chunk['message'])

        self.request_token = chunk['usage']['input_tokens']
        self.response_token = chunk['usage']['output_tokens']

        content = chunk['output']['choices'][0]['message']['content']
        chunk = ChatCompletion(
            id=chunk['request_id'],
            object='',
            model=self.source_model,
            choices=[ChatChoice(
                index=0,
                message={
                    'content': content[self.index:],
                    'role': 'assistant'

                },
            )],
            created=self.created,
        )
        self.index = len(content)

        self.response_text += chunk.choices[0].message.get('content') or ''
        return chunk

    def convert_embedding_response(self, chunk: str) -> Optional[EmbeddingResponse]:
        if isinstance(chunk, str) and not chunk.strip():
            return None

        chunk = orjson.loads(chunk)

        if 'code' in chunk:
            raise OtherException(detail=chunk['message'])

        self.request_token = chunk['usage']['total_tokens']
        self.response_token = 0

        chunk = EmbeddingResponse(
            data=[
                EmbeddingData(
                    index=0,
                    embedding=data['embedding'],
                    object='')
                for data in chunk['output']['embeddings']
            ],
            model=self.source_model,
        )

        if self.encoding_format == 'base64':
            for d in chunk.data:
                d.embedding = encode_embedding(d.embedding)

        return chunk

    def convert_response(self, chunk: str) -> Optional[APIResponse]:
        if self.request_type == RequestType.chat:
            if self.is_stream:
                return self.convert_chat_stream_response(chunk)
            else:
                return self.convert_chat_response(chunk)
        elif self.request_type == RequestType.embeddings:
            return self.convert_embedding_response(chunk)
