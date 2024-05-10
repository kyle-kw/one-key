# -*- coding: utf-8 -*-

# @Time    : 2024/4/16 14:48
# @Author  : kewei

import time
import httpx
import orjson
from loguru import logger
from typing import Optional
from app.relay.base import RelayBase
from app.schemas.relay import ChatBody, APIResponse, StreamChatCompletion, \
    ChatCompletion, APIBody, ChatChoice, StreamChatChoice
from app.utils.common import strip_prefix
from app.utils.crud.models import ChannelKey
from app.utils.exceptions import OtherException


class AnthropicRelay(RelayBase):
    def __init__(self):
        super().__init__()
        self.is_stream = False
        self.res_id: str = ''
        self.res_model: str = ''

    def get_model_list(self):
        return {'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'}

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

        if chat_body.max_tokens:
            max_tokens = min(max(chat_body.max_tokens, 100), 4096)
        else:
            max_tokens = 1024

        system_message = None
        messages = chat_body.messages
        if messages[0]['role'] == 'system':
            system_message = messages[0]['content']
            messages = messages[1:]

        json_data = {
            'model': api_model,
            'messages': messages,
            'max_tokens': max_tokens,
        }

        if system_message:
            json_data['system'] = system_message

        if chat_body.stop:
            json_data['stop_sequences'] = chat_body.stop

        if chat_body.stream:
            json_data['stream'] = chat_body.stream

        if chat_body.temperature:
            json_data['temperature'] = chat_body.temperature

        if chat_body.top_p:
            json_data['top_p'] = chat_body.top_p

        if chat_body.user:
            json_data['metadata'] = {}
            json_data['metadata']['user_id'] = chat_body.user

        url = api_base

        headers = {
            'content-type': 'application/json',
            'anthropic-version': '2023-06-01',
            'x-api-key': api_key,
        }

        request = httpx.Request('POST', url,
                                headers=headers,
                                json=json_data)

        self.request_message = '\n'.join(d['content'] for d in chat_body.messages)
        self.is_stream = chat_body.stream
        self.model = api_model
        self.key_type = key.key_type
        self.source_model = chat_body.model

        logger.info(f"AnthropicRelay.convert_request: {request}")

        return request

    def convert_chat_stream_response(self, chunk: str) -> Optional[StreamChatCompletion]:
        if not chunk.strip() or chunk.startswith('event'):
            return None

        chunk = strip_prefix(chunk, 'data:')
        chunk = orjson.loads(chunk)
        if chunk['type'] == 'error':
            raise OtherException(str(chunk['error']))

        if chunk['type'] == 'message_start':
            self.res_id = chunk['message']['id']
            self.res_model = chunk['message']['model']
            self.request_token = chunk['message']['usage']['input_tokens']
            return None

        if chunk['type'] == 'message_delta':
            self.response_token = chunk['usage']['output_tokens']
            return None

        if chunk['type'] in ['content_block_start', 'ping', 'content_block_stop', 'message_stop']:
            return None

        stream_chunk = StreamChatCompletion(
            id=self.res_id,
            object='',
            model=self.res_model,
            choices=[StreamChatChoice(
                index=0,
                delta={'role': 'assistant', 'content': chunk['delta']['text']}
            )],
            created=int(time.time()),
        )
        self.response_text += stream_chunk.choices[0].delta.get('content') or ''
        return stream_chunk

    def convert_chat_response(self, chunk: str) -> Optional[ChatCompletion]:

        if isinstance(chunk, str) and not chunk.strip():
            return None

        chunk = orjson.loads(chunk)
        chunk = ChatCompletion(
            id=chunk.get('id') or '',
            object='',
            model=self.source_model,
            choices=[ChatChoice(
                index=0,
                message={'role': 'assistant', 'content': chunk['content'][0]['text']},
            )],
            created=int(time.time()),
            usage=chunk.get('usage') or {},
        )
        self.response_text += chunk.choices[0].message.get('content') or ''
        return chunk

    def convert_response(self, chunk: str) -> Optional[APIResponse]:
        if self.is_stream:
            return self.convert_chat_stream_response(chunk)
        else:
            return self.convert_chat_response(chunk)
