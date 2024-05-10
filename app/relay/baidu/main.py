# -*- coding: utf-8 -*-

# @Time    : 2024/4/11 16:58
# @Author  : kewei

import time
import httpx
import orjson
from loguru import logger
from typing import Optional
from cachetools import cached, TTLCache
from app.relay.base import RelayBase
from app.schemas.relay import ChatBody, APIResponse, StreamChatCompletion, \
    ChatCompletion, ChatChoice, StreamChatChoice, APIBody, EmbeddingBody, EmbeddingResponse, EmbeddingData
from app.utils.common import strip_prefix, encode_embedding
from app.utils.exceptions import RequestException
from app.utils.crud.models import ChannelKey
from app.relay.schemas import RequestType


@cached(cache=TTLCache(maxsize=10, ttl=30))
def get_access_token(client_id, client_secret) -> str:
    """
    使用 API Key，Secret Key 获取access_token
    """

    url = "https://aip.baidubce.com/oauth/2.0/token"

    params = {
        'grant_type': 'client_credentials',
        'client_id': client_id,  # API Key
        'client_secret': client_secret,  # Secret Key
    }

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = httpx.request("POST", url, headers=headers, params=params)

    return response.json().get("access_token")


# https://cloud.baidu.com/doc/WENXINWORKSHOP/s/jlil56u11
class BaiduRelay(RelayBase):
    def __init__(self):
        super().__init__()
        self.is_stream = False
        self.request_message: str = ''
        self.response_text: str = ''
        self.model: str = ''
        self.request_type: RequestType = RequestType.chat
        self.encoding_format: Optional[str] = None

    def get_model_list(self):
        return {'ERNIE-4.0-8K', 'ERNIE-4.0-8K-Preview', 'ERNIE-3.5-8K', 'ERNIE-Bot-8K', 'ERNIE-3.5-8K-0205',
                'ERNIE-3.5-8K-1222', 'ERNIE-3.5-4K-0205', 'ERNIE-3.5-8K-Preview',
                'ERNIE-Speed-8K', 'ERNIE-Lite-8K-0922', 'ERNIE-Lite-8K-0308', 'ERNIE-Tiny-8K',
                'ERNIE-Character-8K-0321',
                'Embedding-V1', 'bge-large-en', 'tao-8k'}  # , 'bge-large-zh'

    def get_baidu_url_path(self, model: str) -> str:
        if model == 'ERNIE-4.0-8K':
            return '/chat/completions_pro'
        elif model == 'ERNIE-4.0-8K-Preview':
            return '/chat/ernie-4.0-8k-preview'
        elif model == 'ERNIE-3.5-8K':
            return '/chat/completions'
        elif model == 'ERNIE-Bot-8K':
            return '/chat/ernie_bot_8k'
        elif model == 'ERNIE-3.5-8K-0205':
            return '/chat/ernie-3.5-8k-0205'
        elif model == 'ERNIE-3.5-8K-1222':
            return '/chat/ernie-3.5-8k-1222'
        elif model == 'ERNIE-3.5-4K-0205':
            return '/chat/ernie-3.5-4k-0205'
        elif model == 'ERNIE-3.5-8K-Preview':
            return '/chat/ernie-3.5-8k-preview'
        elif model == 'ERNIE-Speed-8K':
            return '/chat/ernie_speed'
        elif model == 'ERNIE-Lite-8K-0922':
            return '/chat/eb-instant'
        elif model == 'ERNIE-Lite-8K-0308':
            return '/chat/ernie-lite-8k'
        elif model == 'ERNIE-Tiny-8K':
            return '/chat/ernie-tiny-8k'
        elif model == 'ERNIE-Character-8K-0321':
            return '/chat/ernie-char-8k'
        elif model == 'Embedding-V1':
            return '/embeddings/embedding-v1'
        elif model == 'bge-large-zh':
            return '/embeddings/bge_large_zh'
        elif model == 'bge-large-en':
            return '/embeddings/bge_large_en'
        elif model == 'tao-8k':
            return '/embeddings/tao_8k'
        elif model == 'ERNIE-Bot-turbo':
            return '/chat/eb-instant'
        elif model == 'ERNIE-Bot':
            return '/chat/completions'
        elif model == 'BLOOMZ-7B':
            return '/chat/bloomz_7b1'

        raise RequestException(detail='baidu model类型错误！')

    def deal_chat_messages(self, messages) -> list:
        messages_new = []
        for msg in messages:
            if not messages_new:
                messages_new.append(msg)
                continue
            if msg['role'] == messages_new[-1]['role']:
                messages_new[-1]['content'] += msg['content']
            else:
                messages_new.append(msg)

        if len(messages_new) % 2 == 0:
            raise RequestException(detail='baidu messages个数必须为奇数。')

        return messages_new

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

        url_path = self.get_baidu_url_path(api_model)
        url = api_base + url_path

        client_id, client_secret = api_key.split('/')
        access_token = get_access_token(client_id, client_secret)

        params = {
            'access_token': access_token
        }
        headers = {
            'Content-Type': 'application/json',
        }
        system_text = None
        if chat_body.messages[0]['role'] == 'system':
            system_text = chat_body.messages.pop(0).get('content') or ''
        json_data = {
            'messages': self.deal_chat_messages(chat_body.messages),
        }
        if system_text:
            json_data['system'] = system_text

        if chat_body.temperature:
            json_data['temperature'] = min(max(chat_body.temperature, 0.01), 1)

        if chat_body.top_p:
            json_data['top_p'] = min(max(chat_body.top_p, 0), 1)

        if chat_body.frequency_penalty:
            json_data['penalty_score'] = min(max(chat_body.presence_penalty, 1), 2)

        if chat_body.stop:
            json_data['stop'] = chat_body.stop

        if chat_body.max_tokens:
            json_data['max_output_tokens'] = min(max(chat_body.max_tokens, 2), 2048)

        if chat_body.user:
            json_data['user_id'] = chat_body.user

        if chat_body.stream:
            json_data['stream'] = chat_body.stream

        request = httpx.Request('POST', url,
                                headers=headers,
                                json=json_data,
                                params=params)

        self.request_message = '\n'.join(d['content'] for d in chat_body.messages)
        self.is_stream = chat_body.stream
        self.source_model = chat_body.model
        self.model = api_model
        self.key_type = key.key_type

        logger.info(f"BaiduRelay.convert_request: {request}")

        return request

    def build_embedding_request(self, embedding_body: EmbeddingBody, key: ChannelKey) -> httpx.Request:

        api_key = key.api_key
        api_base = key.api_base
        api_model = key.api_model if key.api_model != '*' else embedding_body.model

        texts = (
            [embedding_body.input]
            if isinstance(embedding_body.input, str)
            else embedding_body.input
        )
        if len(texts) > 16:
            raise RequestException(detail='baidu embedding input个数不能超过16个。')

        self.model = api_model

        url_path = self.get_baidu_url_path(api_model)
        url = api_base + url_path

        client_id, client_secret = api_key.split('/')
        access_token = get_access_token(client_id, client_secret)
        params = {
            'access_token': access_token
        }

        headers = {
            'Content-Type': 'application/json',
        }

        json_data = {
            "input": texts
        }

        if embedding_body.user:
            json_data['user_id'] = embedding_body.user

        request = httpx.Request('POST', url,
                                headers=headers,
                                json=json_data,
                                params=params)

        self.request_message = '\n'.join(texts)
        self.is_stream = False
        self.model = api_model
        self.key_type = key.key_type
        self.encoding_format = embedding_body.encoding_format
        self.source_model = embedding_body.model

        logger.info(f"BaiduRelay.convert_request embedding: {request}")

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
        if not chunk.strip():
            return None

        chunk = strip_prefix(chunk, 'data:')
        chunk = orjson.loads(chunk)
        if 'error_code' in chunk:
            raise RequestException(detail=chunk['error_msg'])
        stream_chunk = StreamChatCompletion(
            id=chunk.get('id') or '',
            object=chunk.get('object') or '',
            model=self.source_model,
            choices=[StreamChatChoice(
                index=0,
                delta={'role': 'assistant', 'content': chunk['result']},
                finish_reason=chunk.get('finish_reason') or '',
            )],
            created=chunk.get('created') or int(time.time()),
        )
        self.response_text += stream_chunk.choices[0].delta.get('content') or ''
        return stream_chunk

    def convert_chat_response(self, chunk: str) -> Optional[ChatCompletion]:
        if not chunk.strip():
            return None

        chunk = orjson.loads(chunk)
        if 'error_code' in chunk:
            raise RequestException(detail=chunk['error_msg'])

        chunk = ChatCompletion(
            id=chunk.get('id') or '',
            object=chunk.get('object') or '',
            model=self.source_model,
            choices=[ChatChoice(
                index=0,
                message={'role': 'assistant', 'content': chunk['result']}
            )],
            created=chunk.get('created') or int(time.time()),
        )
        self.response_text += chunk.choices[0].message.get('content') or ''
        return chunk

    def convert_embedding_response(self, chunk: str) -> Optional[EmbeddingResponse]:
        if isinstance(chunk, str) and not chunk.strip():
            return None

        chunk = orjson.loads(chunk)
        if 'error_code' in chunk:
            raise RequestException(detail=chunk['error_msg'])

        chunk = EmbeddingResponse(
            object=chunk['object'],
            model=self.source_model,
            data=[EmbeddingData(
                index=d['index'],
                embedding=d['embedding'],
                object=d['object']
            ) for d in chunk['data']],
            usage=chunk['usage'],
        )

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
