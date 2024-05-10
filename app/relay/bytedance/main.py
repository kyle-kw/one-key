# -*- coding: utf-8 -*-

# @Time    : 2024/4/28 13:08
# @Author  : kewei

import json
import datetime
import hashlib
import hmac
import httpx
import orjson
import time
from enum import Enum
from loguru import logger
from typing import Optional

from app.relay.base import RelayBase
from app.schemas.relay import ChatBody, APIResponse, StreamChatCompletion, \
    ChatCompletion, APIBody, ChatChoice, StreamChatChoice, StreamEndChatCompletion, EmbeddingBody, \
    EmbeddingResponse, EmbeddingData
from app.utils.common import strip_prefix, convert_input_data, encode_embedding
from app.utils.crud.models import ChannelKey
from app.utils.exceptions import RequestException
from app.relay.schemas import RequestType


# sha256 非对称加密
def hmac_sha256(key: bytes, content: str) -> bytes:
    return hmac.new(key, content.encode("utf-8"), hashlib.sha256).digest()


# sha256 hash算法
def hash_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class APIType(str, Enum):
    chat: str = 'chat'
    embedding: str = 'embedding'


def build_request(ak: str, sk: str, body: dict, api_type: APIType = APIType.chat) -> httpx.Request:
    # 创建身份证明。其中的 Service 和 Region 字段是固定的。ak 和 sk 分别代表
    # AccessKeyID 和 SecretAccessKey。同时需要初始化签名结构体。一些签名计算时需要的属性也在这里处理。
    # 初始化身份证明结构体
    credential = {
        "access_key_id": ak,
        "secret_access_key": sk,
        "service": 'ml_maas',
        "region": 'cn-beijing',
    }

    # 初始化签名结构体
    request_param = {
        "body": json.dumps(body) if body else '',
        "host": 'maas-api.ml-platform-cn-beijing.volces.com',
        "path": "/api/v1/chat",
        "method": 'POST',
        "content_type": 'application/json',
        "date": datetime.datetime.utcnow(),
    }

    if api_type == APIType.embedding:
        credential['service'] = 'air'
        request_param['host'] = 'api-vikingdb.volces.com'
        request_param['path'] = '/api/data/embedding/version/2'

    # 接下来开始计算签名。在计算签名前，先准备好用于接收签算结果的 signResult 变量，并设置一些参数。
    # 初始化签名结果的结构体
    x_date = request_param["date"].strftime("%Y%m%dT%H%M%SZ")
    short_x_date = x_date[:8]
    x_content_sha256 = hash_sha256(request_param["body"])
    sign_result = {
        'Accept': 'application/json',
        "Host": request_param["host"],
        "X-Content-Sha256": x_content_sha256,
        "X-Date": x_date,
        "Content-Type": request_param["content_type"],
    }
    # 第五步：计算 Signature 签名。
    signed_headers_str = ";".join(
        ["content-type", "host", "x-content-sha256", "x-date"]
    )

    canonical_request_str = "\n".join(
        [request_param["method"].upper(),
         request_param["path"],
         '',  # query
         "\n".join(
             [
                 "content-type:" + request_param["content_type"],
                 "host:" + request_param["host"],
                 "x-content-sha256:" + x_content_sha256,
                 "x-date:" + x_date,
             ]
         ),
         "",
         signed_headers_str,
         x_content_sha256,
         ]
    )

    hashed_canonical_request = hash_sha256(canonical_request_str)

    credential_scope = "/".join([short_x_date, credential["region"], credential["service"], "request"])
    string_to_sign = "\n".join(["HMAC-SHA256", x_date, credential_scope, hashed_canonical_request])

    k_date = hmac_sha256(credential["secret_access_key"].encode("utf-8"), short_x_date)
    k_region = hmac_sha256(k_date, credential["region"])
    k_service = hmac_sha256(k_region, credential["service"])
    k_signing = hmac_sha256(k_service, "request")
    signature = hmac_sha256(k_signing, string_to_sign).hex()

    sign_result["Authorization"] = "HMAC-SHA256 Credential={}, SignedHeaders={}, Signature={}".format(
        credential["access_key_id"] + "/" + credential_scope,
        signed_headers_str,
        signature,
    )

    # 构建request对象
    r = httpx.Request(method='POST',
                      url="https://{}{}".format(request_param["host"], request_param["path"]),
                      headers=sign_result,
                      json=body,
                      )
    return r


def get_model_max_token(model_name):
    if model_name.endswith('4k'):
        return 4000
    elif model_name.endswith('8k'):
        return 8000
    elif model_name.endswith('16k'):
        return 16000
    elif model_name.endswith('32k'):
        return 32000
    elif model_name == 'skylark-lite':
        return 8000
    elif model_name in ['skylark-plus', 'skylark-chat']:
        return 2000
    elif model_name == 'skylark-pro':
        return 4000
    return 4000


# https://www.volcengine.com/docs/82379/1206178#api-specification
class ByteDanceRelay(RelayBase):
    def __init__(self):
        super().__init__()
        self.is_stream = False
        self.request_message: str = ''
        self.response_text: str = ''
        self.request_type: RequestType = RequestType.chat
        self.encoding_format: Optional[str] = None

    def get_model_list(self):
        return {'skylark2-lite-8k', 'skylark2-pro-32k', 'skylark2-pro-4k', 'skylark2-pro-character-4k',
                'skylark2-pro-turbo-8k', 'skylark-lite', 'skylark-plus', 'skylark-chat', 'skylark-pro',
                'bge-large-zh', 'bge-m3', 'bge-large-zh-and-m3'}

    def build_chat_request(self, chat_body: ChatBody, key: ChannelKey) -> httpx.Request:
        """
        格式化请求，构造httpx.Request
        实例化self.request_message, self.send_request
        """

        api_key = key.api_key
        ak, sk = api_key.split('/')

        api_model = key.api_model if key.api_model != '*' else chat_body.model

        json_data = {
            "model": {
                "name": api_model,
            },
            "parameters": {},
            "messages": chat_body.messages,
        }

        if chat_body.stream:
            json_data["stream"] = chat_body.stream

        if chat_body.max_tokens:
            json_data["parameters"]["max_new_tokens"] = min(max(chat_body.max_tokens, 1),
                                                            get_model_max_token(api_model))

        if chat_body.temperature:
            json_data["parameters"]["temperature"] = min(max(chat_body.temperature, 0.01), 1)

        if chat_body.top_p:
            json_data["parameters"]["top_p"] = min(max(chat_body.top_p, 0), 1)

        request = build_request(ak, sk, json_data)

        self.request_message = '\n'.join(d['content'] for d in chat_body.messages)
        self.is_stream = chat_body.stream
        self.source_model = chat_body.model
        self.model = api_model
        self.key_type = key.key_type

        logger.info(f"ByteDanceRelay.convert_request: {request}")

        return request

    def build_embedding_request(self, embedding_body: EmbeddingBody, key: ChannelKey) -> httpx.Request:

        api_key = key.api_key
        ak, sk = api_key.split('/')
        api_model = key.api_model if key.api_model != '*' else embedding_body.model

        texts = [
            {'data_type': 'text', 'text': d}
            for d in convert_input_data(embedding_body.input)
        ]
        if len(texts) > 100:
            raise RequestException(detail='ByteDance embedding input个数不能超过100个。')

        json_data = {
            "model": {"model_name": api_model},
            "data": texts
        }
        request = build_request(ak, sk, json_data, api_type=APIType.embedding)

        self.request_message = '\n'.join(d['text'] for d in texts)
        self.is_stream = False
        self.model = api_model
        self.key_type = key.key_type
        self.encoding_format = embedding_body.encoding_format
        self.source_model = embedding_body.model

        logger.info(f"ByteDanceRelay.convert_request embedding: {request}")

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
        print(chunk)
        if not chunk.strip() or 'data:[DONE]' == chunk or 'data: [DONE]' == chunk:
            return None

        chunk = strip_prefix(chunk, 'data:')
        chunk = orjson.loads(chunk)

        if 'error' in chunk:
            raise RequestException(detail=chunk['error']['message'])

        if chunk['choice'].get('finish_reason') == 'stop':
            stream_chunk = StreamEndChatCompletion(
                id=chunk['req_id'],
                object='',
                model=self.source_model,
                choices=[StreamChatChoice(
                    index=0,
                    delta={
                        'content': chunk['choice']['message']['content'],
                        'role': 'assistant',
                    },
                    finish_reason='stop',
                )],
                created=int(time.time()),
                usage=chunk['usage']
            )
            self.request_token = stream_chunk.usage.get('prompt_tokens')
            self.response_token = stream_chunk.usage.get('completion_tokens')
        else:
            stream_chunk = StreamChatCompletion(
                id=chunk['req_id'],
                object='',
                model=self.source_model,
                choices=[StreamChatChoice(
                    index=0,
                    delta={
                        'content': chunk['choice']['message']['content'],
                        'role': 'assistant',
                    },
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
            id=chunk['req_id'],
            object='',
            model=self.source_model,
            choices=[ChatChoice(
                index=0,
                message={
                    'role': 'assistant',
                    'content': chunk['choice']['message']['content']
                },
                finish_reason='stop'
            )],
            created=int(time.time()),
            usage=chunk['usage']
        )
        self.response_text += chunk.choices[0].message.get('content') or ''
        self.request_token = chunk.usage.get('prompt_tokens')
        self.response_token = chunk.usage.get('completion_tokens')
        return chunk

    def convert_embedding_response(self, chunk: str) -> Optional[EmbeddingResponse]:
        if isinstance(chunk, str) and not chunk.strip():
            return None

        chunk = orjson.loads(chunk)
        if chunk.get('code') != 0:
            raise RequestException(detail=chunk.get('message', '请求错误'))

        chunk = EmbeddingResponse(
            object='',
            model=self.source_model,
            data=[EmbeddingData(
                index=0,
                embedding=d,
                object=''
            ) for d in chunk['data']['sentence_dense_embedding']],
            usage=chunk['data'].get('token_usage'),
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
