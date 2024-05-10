# -*- coding: utf-8 -*-

import httpx
from loguru import logger
from typing import Optional
from app.relay.base import RelayBase
from app.schemas.relay import ChatBody, APIBody, EmbeddingBody, APIResponse, StreamChatCompletion, \
    ChatCompletion, EmbeddingResponse, StreamEndChatCompletion
from app.utils.common import strip_prefix, count_text_tokens, convert_input_data, convert_embedding_data
from app.relay.schemas import RequestType
from app.utils.crud.models import ChannelKey


class OpenaiRelay(RelayBase):
    def __init__(self):
        super().__init__()
        self.is_stream = False
        self.request_type: RequestType = RequestType.chat
        self.encoding_format: Optional[str] = None

    def get_model_list(self):
        return {'whisper-1', 'davinci-002', 'dall-e-2', 'tts-1-hd-1106', 'tts-1-hd', 'gpt-3.5-turbo',
                'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-instruct-0914', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-16k',
                'gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-0613', 'tts-1', 'dall-e-3',
                'gpt-3.5-turbo-1106', 'babbage-002', 'gpt-4-0125-preview', 'gpt-4-turbo-preview', 'tts-1-1106',
                'text-embedding-3-large', 'gpt-4-turbo-2024-04-09', 'gpt-4-vision-preview', 'text-embedding-3-small',
                'gpt-4', 'text-embedding-ada-002', 'gpt-4-1106-vision-preview', 'gpt-4-1106-preview',
                'gpt-4-0613', 'gpt-4-turbo'}

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
        api_model = key.api_model
        api_config = key.api_config or {}

        api_version = api_config.get('api_version')
        new_model = api_config.get('model')

        json_data = chat_body.dict(exclude_defaults=True)
        if new_model:
            json_data['model'] = new_model
        elif api_model != '*':
            json_data['model'] = api_model

        url = api_base + '/chat/completions'

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'api-key': api_key
        }

        params = None
        if api_version:
            params = {
                'api-version': api_version
            }

        request = httpx.Request('POST', url,
                                headers=headers,
                                json=json_data,
                                params=params)

        self.request_message = '\n'.join(d['content'] for d in chat_body.messages)
        self.is_stream = chat_body.stream
        self.source_model = chat_body.model
        self.model = json_data['model']
        self.key_type = key.key_type

        logger.info(f"OpenaiRelay.convert_request chat: {request}")

        return request

    def build_embedding_request(self, embedding_body: EmbeddingBody, key: ChannelKey) -> httpx.Request:

        api_key = key.api_key
        api_base = key.api_base
        api_model = key.api_model
        api_config = key.api_config or {}
        api_version = api_config.get('api_version')

        json_data = embedding_body.dict(exclude_defaults=True)
        if api_model != '*':
            json_data['model'] = api_model

        json_data['input'] = convert_input_data(json_data['input'])

        url = api_base + '/embeddings'

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'api-key': api_key
        }
        params = None
        if api_version:
            params = {
                'api-version': api_version
            }
        request = httpx.Request('POST', url,
                                headers=headers,
                                json=json_data,
                                params=params)

        self.request_message = '\n'.join(json_data['input'])
        self.is_stream = False
        self.model = api_model
        self.key_type = key.key_type
        self.source_model = embedding_body.model
        self.encoding_format = embedding_body.encoding_format

        logger.info(f"OpenaiRelay.convert_request embedding: {request}")
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
        if not chunk.strip() or 'data:[DONE]' == chunk or 'data: [DONE]' == chunk:
            return None

        chunk = strip_prefix(chunk, 'data:')

        stream_chunk = StreamChatCompletion.parse_raw(chunk)

        if stream_chunk.choices and stream_chunk.choices[0].finish_reason == 'stop':
            stream_chunk = StreamEndChatCompletion.parse_raw(chunk)
            if stream_chunk.choices[0].delta.get('role') != 'assistant':
                stream_chunk.choices[0].delta['role'] = 'assistant'

            if not stream_chunk.usage:
                stream_chunk.usage = {}
                stream_chunk.usage['prompt_tokens'] = count_text_tokens(self.request_message)
                stream_chunk.usage['completion_tokens'] = count_text_tokens(self.response_text)
                stream_chunk.usage['total_tokens'] = stream_chunk.usage['prompt_tokens'] + stream_chunk.usage['completion_tokens']

            self.request_token = stream_chunk.usage.get('prompt_tokens')
            self.response_token = stream_chunk.usage.get('completion_tokens')
            self.response_text += stream_chunk.choices[0].delta.get('content') or ''

        elif stream_chunk.choices and stream_chunk.choices[0].delta.get('content'):
            self.response_text += stream_chunk.choices[0].delta.get('content') or ''

            if stream_chunk.choices[0].delta.get('role') != 'assistant':
                stream_chunk.choices[0].delta['role'] = 'assistant'

        else:
            return None

        stream_chunk.model = self.source_model

        return stream_chunk

    def convert_chat_response(self, chunk: str) -> Optional[ChatCompletion]:

        if isinstance(chunk, str) and not chunk.strip():
            return None

        chunk = ChatCompletion.parse_raw(chunk)

        chunk.model = self.source_model
        if chunk.choices and chunk.choices[0].message.get('content'):
            self.response_text += chunk.choices[0].message.get('content') or ''

            if chunk.choices[0].message.get('role') != 'assistant':
                chunk.choices[0].message['role'] = 'assistant'
        else:
            return None

        if not chunk.usage:
            chunk.usage = {}
            chunk.usage['prompt_tokens'] = count_text_tokens(self.request_message)
            chunk.usage['completion_tokens'] = count_text_tokens(self.response_text)
            chunk.usage['total_tokens'] = chunk.usage['prompt_tokens'] + chunk.usage['completion_tokens']

        self.request_token = chunk.usage.get('prompt_tokens')
        self.response_token = chunk.usage.get('completion_tokens')

        return chunk

    def convert_embedding_response(self, chunk: str) -> Optional[EmbeddingResponse]:
        if isinstance(chunk, str) and not chunk.strip():
            return None

        chunk = EmbeddingResponse.parse_raw(chunk)
        chunk.model = self.source_model
        for embedding_data in chunk.data:
            embedding_data.embedding = convert_embedding_data(embedding_data.embedding, self.encoding_format)

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
