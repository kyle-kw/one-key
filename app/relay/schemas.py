# -*- coding: utf-8 -*-

# @Time    : 2024/4/11 13:38
# @Author  : kewei

from enum import Enum


class RelayType(str, Enum):
    ali = 'ali'
    ai360 = 'ai360'
    anthropic = 'anthropic'
    baichuan = 'baichuan'
    baidu = 'baidu'
    gemini = 'gemini'
    moonshot = 'moonshot'
    openai = 'openai'
    tencent = 'tencent'
    xunfei = 'xunfei'
    zhipu = 'zhipu'
    huawei = 'huawei'
    bytedance = 'bytedance'


class RequestType(str, Enum):
    chat = 'chat'
    embeddings = 'embeddings'
