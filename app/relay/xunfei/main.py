# -*- coding: utf-8 -*-

# @Time    : 2024/4/15 10:12
# @Author  : kewei

import time
import httpx
import orjson
import websockets
import base64
import datetime
import hashlib
import hmac
from loguru import logger
from typing import Optional, AsyncIterator
from urllib.parse import urlparse
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

from cachetools import cached, TTLCache

from app.relay.base import RelayBase
from app.schemas.relay import ChatBody, APIResponse, StreamChatCompletion, \
    ChatCompletion, ChatChoice, StreamChatChoice, APIBody
from app.utils.exceptions import OtherException
from app.utils.crud.models import ChannelKey, AuthenticationKey


@cached(cache=TTLCache(maxsize=10, ttl=30))
def generate_ws_url(api_key, api_secret, spark_url):
    parse_url = urlparse(spark_url)
    spark_host = parse_url.netloc
    spark_path = parse_url.path

    # 生成RFC1123格式的时间戳
    now = datetime.now()
    date = format_date_time(mktime(now.timetuple()))

    # 拼接字符串
    signature_origin = (
        f"host: {spark_host}\n"
        f"date: {date}\n"
        f"GET {spark_path} HTTP/1.1"
    )

    # 进行hmac-sha256进行加密
    signature_sha = hmac.new(api_secret.encode('utf-8'),
                             signature_origin.encode('utf-8'),
                             digestmod=hashlib.sha256).digest()

    signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

    authorization_origin = (
        f'api_key="{api_key}", '
        'algorithm="hmac-sha256", '
        'headers="host date request-line", '
        f'signature="{signature_sha_base64}"'
    )

    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

    # 将请求的鉴权参数组合为字典
    v = {
        "authorization": authorization,
        "date": date,
        "host": spark_host
    }
    # 拼接鉴权参数，生成url
    url = spark_url + '?' + urlencode(v)

    return url


# https://www.xfyun.cn/doc/spark/Web.html
class XunfeiRelay(RelayBase):
    def __init__(self):
        super().__init__()
        self.is_stream = False
        self.request_message: str = ''
        self.response_text: str = ''
        self.model: str = ''
        self.no_stream_model = None

    def get_model_list(self):
        return {'generalv3.5', 'generalv3', 'generalv2', 'general'}

    def get_xunfei_url_path(self, model: str) -> str:
        if model == 'generalv3.5':
            return 'wss://spark-api.xf-yun.com/v3.5/chat'
        elif model == 'generalv3':
            return 'wss://spark-api.xf-yun.com/v3.1/chat'
        elif model == 'generalv2':
            return 'wss://spark-api.xf-yun.com/v2.1/chat'
        elif model == 'general':
            return 'wss://spark-api.xf-yun.com/v1.1/chat'

    def get_xunfei_max_tokens(self, model: str) -> int:
        if model == 'general':
            return 4096
        else:
            return 8192

    def build_request(self, chat_body: ChatBody, key: ChannelKey) -> httpx.Request:
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

        spark_url = api_base or self.get_xunfei_url_path(api_model)

        app_id, api_secret, api_key_ = api_key.split('/')
        ws_url = generate_ws_url(api_key_, api_secret, spark_url)
        json_data = {
            'header': {
                'app_id': app_id,
            },
            'parameter': {
                'chat': {
                    'domain': api_model,
                }
            },
            'payload': {
                'message': {
                    'text': chat_body.messages
                }
            }
        }

        if chat_body.user:
            json_data['header']['uid'] = chat_body.user

        if chat_body.max_tokens:
            json_data['parameter']['chat']['max_tokens'] = min(max(chat_body.max_tokens, 1),
                                                               self.get_xunfei_max_tokens(api_model))

        if chat_body.temperature:
            json_data['parameter']['chat']['temperature'] = min(max(chat_body.temperature, 0.01), 1)

        if chat_body.n:
            json_data['parameter']['chat']['top_k'] = min(max(chat_body.n, 1), 6)

        request = httpx.Request('POST', ws_url,
                                json=json_data)

        self.request_message = '\n'.join(d['content'] for d in chat_body.messages)
        self.is_stream = chat_body.stream
        self.source_model = chat_body.model

        logger.info(f"XunfeiRelay.convert_request: {request}")

        return request

    async def send_ws_request(self, url: str, data: str) -> AsyncIterator[str]:
        async with websockets.connect(url) as websocket:
            await websocket.send(data)

            while True:
                try:
                    message = await websocket.recv()
                    yield message
                except websockets.ConnectionClosedOK:
                    break
                except websockets.WebSocketException as e:
                    logger.error(e)
                    raise e
                except Exception as e:
                    logger.error(e)
                    raise e

    async def send_request(self, request: httpx.Request) -> AsyncIterator[str]:

        yield_flag = False
        while self.retry_times < 3 and not yield_flag:
            try:
                self.start_time = time.time()
                async for check in self.send_ws_request(str(request.url), request.content.decode()):
                    if not self.first_time:
                        self.first_time = time.time()

                    if not yield_flag:
                        yield_flag = True

                    yield check

                self.all_time = time.time()
                break
            except Exception as e:
                if self.retry_times < 3 and not yield_flag:
                    self.retry_times += 1
                    logger.error(f'执行方法异常，进行重试。{e}')
                    continue
                else:
                    logger.error(f'执行方法异常，已开始返回，直接抛出异常。 {e}')
                    raise e

    def convert_chat_stream_response(self, chunk: str) -> Optional[StreamChatCompletion]:
        if not chunk.strip():
            return None

        chunk = orjson.loads(chunk)

        if chunk['header']['code'] != 0:
            raise OtherException(detail=chunk['header']['message'])

        stream_chunk = StreamChatCompletion(
            id=chunk['header']['sid'],
            object='',
            model=self.source_model,
            choices=[StreamChatChoice(
                index=0,
                delta={
                    'content': chunk['payload']['choices']['text'][0]['content'],
                    'role': 'assistant',
                },
                finish_reason='',
            )],
            created=int(time.time()),
        )
        self.response_text += stream_chunk.choices[0].delta.get('content') or ''
        return stream_chunk

    def convert_chat_response(self, chunk: str) -> Optional[ChatCompletion]:
        if not chunk.strip():
            return None
        chunk = orjson.loads(chunk)

        if chunk['header']['code'] != 0:
            raise OtherException(detail=chunk['header']['message'])

        if not self.no_stream_model:
            chunk = ChatCompletion(
                id=chunk['header']['sid'],
                object='',
                model=self.source_model,
                choices=[ChatChoice(
                    index=0,
                    message={
                        'role': 'assistant',
                        'content': chunk['payload']['choices']['text'][0]['content']
                    }
                )],
                created=int(time.time()),
            )
            self.no_stream_model = chunk
        else:
            self.no_stream_model.choices[0].message['content'] += chunk['payload']['choices']['text'][0]['content']
        self.response_text = self.no_stream_model.choices[0].message.get('content') or ''

        return None

    def convert_response(self, chunk: str) -> Optional[APIResponse]:
        if self.is_stream:
            return self.convert_chat_stream_response(chunk)
        else:
            return self.convert_chat_response(chunk)

    async def main(self, auth_key: AuthenticationKey, api_body: APIBody, key: ChannelKey):
        self.auth_key = auth_key.api_key
        self.auth_key_pk = auth_key.record_pk
        self.source_model = api_body.model
        self.user_id = api_body.user or ''
        self.api_key = key.api_key
        self.api_body = api_body

        request = self.build_request(api_body, key)
        async for chunk in self.send_request(request):
            response = self.convert_response(chunk)
            if not response:
                continue

            yield response

        if not self.is_stream:
            yield self.no_stream_model

        await self.end_response()
