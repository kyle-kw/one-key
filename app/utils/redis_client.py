# -*- coding: utf-8 -*-


import time
import orjson
from redis import Redis, asyncio as aioredis
from redis.exceptions import ConnectionError
from app.utils.exceptions import InternalException
from typing import Optional

add_keys_script = """\
local key = KEYS[1]
local current_time = tonumber(ARGV[1])
redis.call('ZREMRANGEBYSCORE', key, '-inf', current_time)

local elements = cjson.decode(ARGV[1])
return redis.call('ZADD', key, unpack(elements))
"""

get_keys_with_cooldown_script = """\
local key_pool = KEYS[1]
local cooldown_pool = KEYS[2] 
local current_time = tonumber(ARGV[1])
redis.call('ZREMRANGEBYSCORE', cooldown_pool, '-inf', current_time)
redis.call('ZREMRANGEBYSCORE', key_pool, '-inf', current_time)

local keys = redis.call('ZDIFF', 2, key_pool, cooldown_pool)
return keys
"""

get_keys_script = """\
local key_pool = KEYS[1]
local current_time = tonumber(ARGV[1])
redis.call('ZREMRANGEBYSCORE', key_pool, '-inf', current_time)

return redis.call('ZRANGE', key_pool, 0, -1)
"""

add_token_script = """\
local limit_key = KEYS[1]
local now = tonumber(ARGV[1])
local token = tonumber(ARGV[2])
redis.call('hset', limit_key, now, token)
redis.call('expire', limit_key, 120)

return 1
"""

limit_keys_script = """\
local limit_key = KEYS[1]
local now = tonumber(ARGV[1])
local max_token = tonumber(ARGV[2])

local result = redis.call('hgetall', limit_key)

local del_key = {}
local token_cnt = 0
local one_minute_ago = now - 60000
for i = 1, #result, 2 do
    if tonumber(result[i]) < one_minute_ago then
        table.insert(del_key, result[i])
    else
        token_cnt = token_cnt + tonumber(result[i+1])
    end
end

if #del_key ~= 0 then
    redis.call('hdel', limit_key, unpack(del_key))
end

if token_cnt >= max_token then
    return 0
else
    return 1
end
"""


class RedisClient:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.r = None
        self.r: Redis

        self.add_keys_sha = None
        self.get_keys_sha = None
        self.get_keys_with_cooldown_sha = None
        self.add_token_sha = None
        self.limit_keys_sha = None

        self.last_ping = time.time() - 10

    async def client(self):
        init_client = False
        try:
            if time.time() - self.last_ping > 3:
                self.last_ping = time.time()

                if not self.r or not await self.r.ping():
                    init_client = True
        except ConnectionError as e:
            init_client = True

        if init_client:
            self.r = aioredis.from_url(self.redis_url)
            if not await self.r.ping():
                raise InternalException('Redis connection failed')

            await self.init_script_sha()

        return self.r

    async def init_script_sha(self):
        self.add_keys_sha = await self.r.script_load(add_keys_script)
        self.get_keys_sha = await self.r.script_load(get_keys_script)
        self.get_keys_with_cooldown_sha = await self.r.script_load(get_keys_with_cooldown_script)
        self.add_token_sha = await self.r.script_load(add_token_script)
        self.limit_keys_sha = await self.r.script_load(limit_keys_script)

    async def add_keys_to_pool(self, pool_name, keys, timeout=60):
        """
        将keys添加到redis key池
        """
        if not keys:
            return

        await self.client()

        if isinstance(keys, str):
            keys = [keys]

        new_keys = []
        now_time = int(time.time() + timeout)
        for key in keys:
            new_keys.append(now_time)
            new_keys.append(key)
        keys = orjson.dumps(new_keys).decode()

        return await self.r.evalsha(self.add_keys_sha, 1, pool_name, keys)

    async def get_keys_from_pool(self, pool_name, cooldown_name=None):
        """
        从redis key池中获取keys
        """
        await self.client()
        if cooldown_name:
            return await self.r.evalsha(self.get_keys_with_cooldown_sha, 2, pool_name, cooldown_name,
                                        int(time.time()))
        else:
            return await self.r.evalsha(self.get_keys_sha, 1, pool_name, int(time.time()))

    async def limit_keys_token(self, key, max_token):
        await self.client()
        limit_key = 'limit_keys_token_new:' + key
        now_time = int(time.time() * 1000)
        limit_status = await self.r.evalsha(self.limit_keys_sha, 1, limit_key, now_time, max_token)

        return limit_status

    async def add_keys_token(self, key, token):
        await self.client()
        limit_key = 'limit_keys_token_new:' + key
        now_time = int(time.time() * 1000)
        add_status = await self.r.evalsha(self.add_token_sha, 1, limit_key, now_time, token)

        return add_status

    async def client_get(self, key: str):
        await self.client()
        return await self.r.get(key)

    async def client_set(self, key: str, value: bytes, expire: Optional[int] = None):
        await self.client()
        return await self.r.set(key, value, ex=expire)


