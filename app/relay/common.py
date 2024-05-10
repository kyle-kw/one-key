# -*- coding: utf-8 -*-

# from app.relay.schemas import RelayType
from app.relay.openai import OpenaiRelay
from app.relay.ali import AliRelay
from app.relay.baichuan import BaichuanRelay
from app.relay.baidu import BaiduRelay
from app.relay.xunfei import XunfeiRelay
from app.relay.zhipu import ZhipuRelay
from app.relay.moonshot import MoonshotRelay
from app.relay.anthropic import AnthropicRelay
from app.relay.huawei import HuaweiRelay
from app.relay.gemini import GeminiRelay
from app.relay.bytedance import ByteDanceRelay
from app.relay.ai360 import AI360Relay


class ModelMapping:
    def __init__(self):
        self.mapping = {
            'openai': OpenaiRelay().get_model_list(),
            'ali': AliRelay().get_model_list(),
            'baichuan': BaichuanRelay().get_model_list(),
            'baidu': BaiduRelay().get_model_list(),
            'xunfei': XunfeiRelay().get_model_list(),
            'zhipu': ZhipuRelay().get_model_list(),
            'moonshot': MoonshotRelay().get_model_list(),
            'anthropic': AnthropicRelay().get_model_list(),
            'huawei': HuaweiRelay().get_model_list(),
            'gemini': GeminiRelay().get_model_list(),
            'bytedance': ByteDanceRelay().get_model_list(),
            'ai360': AI360Relay().get_model_list(),
        }

    def judge_model_type(self, model_name) -> str:
        for key, mapping in self.mapping.items():
            if model_name in mapping:
                return key
        return ''

    def get_type_model_list(self, model_type):
        return self.mapping.get(model_type, {})


model_mapping = ModelMapping()
judge_model_type = model_mapping.judge_model_type
get_type_model_list = model_mapping.get_type_model_list
