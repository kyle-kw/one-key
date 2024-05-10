import httpx

from app.utils.exceptions import TimeoutException, RequestException, OtherException
from typing import AsyncIterator
from app.utils.logger import logger
from app.config import env_settings


async def _stream_request(request: httpx.Request) -> AsyncIterator[str]:
    async with httpx.AsyncClient(timeout=env_settings.request_timeout) as client:
        try:
            response = await client.send(request, stream=True)
        except httpx.TimeoutException as err:
            logger.debug("Encountered httpx.TimeoutException", exc_info=True)

            raise TimeoutException(detail='request timeout') from err
        except Exception as err:
            logger.debug("Encountered Exception", exc_info=True)

            raise OtherException(detail='network error') from err

        logger.debug('HTTP Request: {} {} "{} {}"'.format(request.method, request.url, response.status_code,
                     response.reason_phrase))

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:  # thrown on 4xx and 5xx status code
            logger.debug("Encountered httpx.HTTPStatusError", exc_info=True)

            content = await response.aread()

            error_message = f"HTTP status error: {response.status_code} {response.reason_phrase} {content.decode()}"
            raise RequestException(detail=error_message) from err

        async for chunk in response.aiter_lines():
            yield chunk

        await response.aclose()


async def _request(request: httpx.Request) -> AsyncIterator[str]:
    async with httpx.AsyncClient(timeout=env_settings.request_timeout) as client:

        try:
            response = await client.send(request)
        except httpx.TimeoutException as err:
            logger.debug("Encountered httpx.TimeoutException", exc_info=True)

            raise TimeoutException(detail='request timeout') from err
        except Exception as err:
            logger.debug("Encountered Exception", exc_info=True)

            raise OtherException(detail='network error') from err

        logger.debug(f'HTTP Request: {request.method} {request.url} "{response.status_code} {response.reason_phrase}"')

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:  # thrown on 4xx and 5xx status code
            logger.debug("Encountered httpx.HTTPStatusError", exc_info=True)

            error_message = f"HTTP status error: {response.status_code} {response.reason_phrase} {response.content.decode()}"
            raise RequestException(detail=error_message) from err

        yield response.content.decode()


def test_req():
    import asyncio
    import httpx
    #api_base = 'https://zhongbiao-iig-009.openai.azure.com/openai/deployments/zhongbiao-gpt-35-turbo-16k'
    #api_key = 'ea961a309100447cbfcaacbcf2e05bba'
    api_base = 'http://localhost:8000/v1/'
    api_base = 'http://10.0.8.13:13390/v1'
    api_base = 'http://172.21.16.15:30522/v1'
    api_key = 'fd-Yy1Zs8sdM2AVz4oHbkhS3p3JOwlVPF8zo9zRiylvvpDuzOz'

    url = api_base + '/chat/completions'
    params = {
        'api-version': '2023-05-15'
    }
    json_data = {
        "model": "gpt-35-turbo-16k",
        "messages": [
            {'role': 'user', 'content': '你好'},
        ],
        "stream": True
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'api-key': api_key
    }
    req: httpx.Request = httpx.Request('POST',
                                       url,
                                       headers=headers,
                                       json=json_data,
                                       params=params)

    async def _test_stream_request():
        res = await _request(req)
        print(res.content.decode())
    asyncio.run(_test_stream_request())


if __name__ == '__main__':
    test_req()
