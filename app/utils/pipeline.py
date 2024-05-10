# -*- coding: utf-8 -*-

# @Time    : 2024/4/24 09:58
# @Author  : kewei

import time
import queue
import redis
import pickle
from collections import defaultdict
from apscheduler.schedulers.background import BackgroundScheduler

from app.utils.logger import logger
from app.config import env_settings
from app.utils.crud.main import insert_record_log
from app.utils.redis_client import add_token_script
from app.utils.crud.models import RecordLog


def object_to_binary(data):
    binary_data = pickle.dumps(data)

    return binary_data


def binary_to_object(binary_data):
    loaded_data = pickle.loads(binary_data)

    return loaded_data


class RedisPipe:
    def __init__(self):
        self.r = redis.from_url(env_settings.redis_url)
        self.queue_name = 'queue:log'
        self.max_size = 1000

    def put(self, data):

        if self.r.llen(self.queue_name) >= self.max_size:
            return False

        data_ = object_to_binary(data)
        self.r.rpush(self.queue_name, data_)
        self.r.expire(self.queue_name, 60)
        return True

    def get(self):
        data = self.r.lpop(self.queue_name)
        data_ = binary_to_object(data) if data else None
        return data_

    def get_batch(self, batch_size=10, timeout=1.0):
        res = []
        now = time.time()
        delay_time = 0
        while len(res) < batch_size and delay_time < timeout:
            data = self.get()
            if data:
                res.append(data)

            time.sleep(0.1)

            delay_time = time.time() - now

        return res

    def add_token(self, key, token):
        limit_key = 'limit_keys_token_new:' + key
        now = int(time.time() * 1000)
        return self.r.eval(add_token_script, 1, limit_key, now, token)


class QueuePipe:
    def __init__(self):
        self.max_size = 1000
        self.q = queue.Queue(maxsize=self.max_size)

    def put(self, data, timeout=0.2):
        try:
            self.q.put(data, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get(self, timeout=1):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_batch(self, batch_size=10, timeout=1.0):
        res = []
        now = time.time()
        delay_time = 0
        while len(res) < batch_size and delay_time < timeout:
            try:
                res.append(self.q.get(timeout=timeout))
            except queue.Empty:
                pass

            time.sleep(0.1)

            delay_time = time.time() - now

        return res

    def add_token(self, key, token):
        pass


class Pipe:
    def __init__(self):
        self.pipe = None
        try:
            self.pipe = RedisPipe()
            self.pipe.r.ping()
        except Exception as e:
            logger.error(f'connect redis error: {e}')
            self.pipe = QueuePipe()

    def get(self):
        return self.pipe.get()

    def put(self, data):
        return self.pipe.put(data)

    def get_batch(self, batch_size=10, timeout=1.0):
        return self.pipe.get_batch(batch_size=batch_size, timeout=timeout)

    def add_token(self, key, token):
        return self.pipe.add_token(key, token)


pipe = Pipe()


def consumer():
    data = pipe.get_batch(timeout=0.5)
    if not data:
        return

    data_format = []
    limit_dict = defaultdict(lambda: 0)
    for d in data:
        data_format.append(RecordLog(**d['record_log']))
        limit_dict[d['auth_key']] += d['all_token']
        limit_dict[d['api_key']] += d['all_token']

    if env_settings.limit_token:
        for key, token in limit_dict.items():
            pipe.add_token(key, token)
        logger.debug(f'add token: {len(limit_dict)}')

    if env_settings.save_log:
        insert_record_log(data_format)
        logger.debug(f'insert data len: {len(data)}')


# 后台定时任务
s = BackgroundScheduler(job_defaults={"misfire_grace_time": 10 * 60})
s.add_job(consumer,
          trigger='interval',
          seconds=1,
          max_instances=1,
          replace_existing=False)
s.start()

