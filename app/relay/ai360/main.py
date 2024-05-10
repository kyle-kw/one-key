# -*- coding: utf-8 -*-

# @Time    : 2024/4/28 13:08
# @Author  : kewei

import httpx
from loguru import logger
from typing import Optional
from app.relay.base import RelayBase
from app.schemas.relay import ChatBody, APIResponse, StreamChatCompletion, \
    ChatCompletion, APIBody
from app.utils.common import strip_prefix
from app.utils.crud.models import ChannelKey


# https://ai.360.com/platform/docs/overview
class AI360Relay(RelayBase):
    def __init__(self):
        super().__init__()
        self.is_stream = False
        self.request_message: str = ''
        self.response_text: str = ''

    def get_model_list(self):
        return {'360gpt-turbo-responsibility-8k', '360gpt-pro-sc202401v3', '360gpt-pro-sc202401v2',
                '360gpt-pro-sc202401v1', '360gpt-pro-v2.0.3', '360GPT_S1_QIYUAN', '360GPT_S2_V9',
                '360gpt-pro', '360gpt-turbo'}

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
        api_model = key.api_model

        json_data = chat_body.dict(exclude_defaults=True)
        if api_model != '*':
            json_data['model'] = api_model

        url = api_base + '/chat/completions'

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        request = httpx.Request('POST', url,
                                headers=headers,
                                json=json_data)

        self.request_message = '\n'.join(d['content'] for d in chat_body.messages)
        self.is_stream = chat_body.stream
        self.source_model = chat_body.model
        self.model = api_model
        self.key_type = key.key_type

        logger.info(f"AI360Relay.convert_request: {request}")

        return request

    def convert_chat_stream_response(self, chunk: str) -> Optional[StreamChatCompletion]:
        if not chunk.strip() or 'data:[DONE]' == chunk or 'data: [DONE]' == chunk:
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

    def convert_response(self, chunk: str) -> Optional[APIResponse]:
        if self.is_stream:
            return self.convert_chat_stream_response(chunk)
        else:
            return self.convert_chat_response(chunk)
