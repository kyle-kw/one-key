# -*- coding: utf-8 -*-

from app.utils.redis_client import RedisClient
from app.config import env_settings

redis_client = RedisClient(redis_url=env_settings.redis_url)

limit_keys_token = redis_client.limit_keys_token
add_keys_token = redis_client.add_keys_token


async def get_openai_key_test():
    key = 'test'
    token = 100
    await add_keys_token(key, token)

    res = await limit_keys_token(key, 1)
    print(res)


if __name__ == '__main__':
    import asyncio

    asyncio.run(get_openai_key_test())
