import time
import httpx
from loguru import logger
from typing import AsyncIterator, Optional
from app.utils.exceptions import TimeoutException, OtherException
from app.relay.request import _stream_request, _request
from app.schemas.relay import APIBody, APIResponse
from app.utils.crud.models import ChannelKey, AuthenticationKey
from app.utils.common import count_text_tokens
from datetime import datetime
from app.utils.pipeline import pipe


class RelayBase:
    def __init__(self):
        self.start_time: int = 0
        self.first_time: int = 0
        self.all_time: int = 0
        self.retry_times: int = 0
        self.is_stream = False
        self.auth_key: str = ''
        self.auth_key_pk: Optional[int] = None

        self.request_message: str = ''
        self.request_token: Optional[int] = None
        self.response_text: str = ''
        self.response_token: Optional[int] = None

        self.key_type: str = ''
        self.model: str = ''
        self.source_model: str = ''
        self.user_id: str = ''
        self.api_key: str = ''

        self.api_body: Optional[APIBody] = None

    def get_model_list(self):
        pass

    def build_request(self, api_body: APIBody, key: ChannelKey) -> httpx.Request:
        raise NotImplementedError

    async def send_request(self, request: httpx.Request) -> AsyncIterator[str]:

        if self.is_stream:
            send_request_func = _stream_request
        else:
            send_request_func = _request

        yield_flag = False
        while self.retry_times < 3 and not yield_flag:
            try:
                self.start_time = time.time()
                async for check in send_request_func(request):
                    if not self.first_time:
                        self.first_time = time.time()

                    if not yield_flag:
                        yield_flag = True

                    yield check

                self.all_time = time.time()
                break
            except Exception as e:
                if not isinstance(e, (TimeoutException, OtherException)):
                    # TODO 添加冷却池
                    raise e

                if self.retry_times < 3 and not yield_flag:
                    self.retry_times += 1
                    logger.error(f'执行方法异常，进行重试。{e}')
                    continue
                else:
                    logger.error(f'执行方法异常，已开始返回，直接抛出异常。 {e}')
                    raise e

    def convert_response(self, chunk: str) -> Optional[APIResponse]:
        raise NotImplementedError

    def record_tokens(self):

        if self.request_token is None:
            self.request_token = count_text_tokens(self.request_message)

        if self.response_token is None:
            self.response_token = count_text_tokens(self.response_text)

    async def end_response(self):
        s = time.time()

        try:
            self.record_tokens()

            record_log = dict(
                auth_key_pk=self.auth_key_pk,
                api_type=self.key_type,
                user_id=self.user_id,
                source_model=self.source_model,
                api_model=self.model,
                prompt=self.request_message,
                completion=self.response_text,
                prompt_token=self.request_token,
                completion_token=self.response_token,
                first_time=round(self.first_time - self.start_time, 2),
                all_time=round(self.all_time - self.start_time, 2),
                return_rate=round(self.response_token / (self.all_time - self.start_time), 2),
                retry_times=self.retry_times,
                api_body=self.api_body.dict(exclude_defaults=True),
                create_time=datetime.now(),
            )

            data = dict(
                record_log=record_log,
                auth_key=self.auth_key,
                api_key=self.api_key,
                all_token=self.request_token + self.response_token,
            )
            put_stats = pipe.put(data)
            logger.debug(f'put_stats: {put_stats}')

        except Exception as e:
            logger.error(f'end_response error: {e}')
        finally:
            logger.debug(f'end_response: {round(time.time() - s, 4)}')

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

        await self.end_response()
