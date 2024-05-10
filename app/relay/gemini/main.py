# -*- coding: utf-8 -*-

# @Time    : 2024/4/16 17:02
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


# https://ai.google.dev/api/rest?hl=zh-cn
class GeminiRelay(RelayBase):
    def __init__(self):
        super().__init__()
        self.is_stream = False

    def get_model_list(self):
        return {'gemini-1.0-pro', 'gemini-1.0-pro-001',
                'gemini-1.0-pro-latest', 'gemini-1.0-pro-vision-latest', 'gemini-1.5-pro-latest', 'gemini-pro',
                'gemini-pro-vision'}

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

        url = api_base + '/models/' + api_model

        params = {
            'key': api_key
        }
        if chat_body.stream:
            url += ':streamGenerateContent'
            params['alt'] = 'sse'
        else:
            url += ':generateContent'

        headers = {
            'content-type': 'application/json',
        }

        messages = chat_body.messages
        if messages[0]['role'] == 'system':
            system_message = messages[0]['content']
            messages = messages[1:]
            messages[0]['content'] = system_message + '\n' + messages[0]['content']

        json_data = {
            'contents': [
                {
                    'role': 'user' if message['role'] == 'user' else 'model',
                    'parts': [
                        {'text': message['content']}
                    ]
                }
                for message in messages
            ],
            'generationConfig': {},
        }

        if chat_body.stop:
            json_data['generationConfig']['stopSequences'] = chat_body.stop

        if chat_body.temperature:
            json_data['generationConfig']['temperature'] = chat_body.temperature

        if chat_body.top_p:
            json_data['generationConfig']['topP'] = chat_body.top_p

        if chat_body.max_tokens:
            json_data['generationConfig']['maxOutputTokens'] = chat_body.max_tokens

        if not json_data['generationConfig']:
            del json_data['generationConfig']

        request = httpx.Request('POST', url,
                                headers=headers,
                                json=json_data,
                                params=params)

        self.request_message = '\n'.join(d['content'] for d in chat_body.messages)
        self.is_stream = chat_body.stream
        self.model = api_model
        self.key_type = key.key_type
        self.source_model = chat_body.model

        logger.info(f"GeminiRelay.convert_request: {request}")

        return request

    def convert_chat_stream_response(self, chunk: str) -> Optional[StreamChatCompletion]:
        if not chunk.strip():
            return None

        chunk = strip_prefix(chunk, 'data:')
        chunk = orjson.loads(chunk)

        stream_chunk = StreamChatCompletion(
            id='',
            object='',
            model=self.source_model,
            choices=[StreamChatChoice(
                index=0,
                delta={'role': 'assistant', 'content': chunk['candidates'][0]['content']['parts'][0]['text']}
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
            id='',
            object='',
            model=self.source_model,
            choices=[ChatChoice(
                index=0,
                message={'role': 'assistant', 'content': chunk['candidates'][0]['content']['parts'][0]['text']},
            )],
            created=int(time.time()),
        )
        self.response_text += chunk.choices[0].message.get('content') or ''
        return chunk

    def convert_response(self, chunk: str) -> Optional[APIResponse]:
        if self.is_stream:
            return self.convert_chat_stream_response(chunk)
        else:
            return self.convert_chat_response(chunk)
