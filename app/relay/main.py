# -*- coding: utf-8 -*-
from app.relay.schemas import RelayType
from app.relay.openai import OpenaiRelay
from app.relay.ali import AliRelay
from app.relay.baichuan import BaichuanRelay
from app.relay.baidu import BaiduRelay
from app.relay.xunfei import XunfeiRelay
from app.relay.zhipu import ZhipuRelay
from app.relay.moonshot import MoonshotRelay
from app.schemas.relay import APIBody
from app.relay.anthropic import AnthropicRelay
from app.relay.huawei import HuaweiRelay
from app.relay.gemini import GeminiRelay
from app.relay.bytedance import ByteDanceRelay
from app.relay.ai360 import AI360Relay
from app.utils.crud.models import ChannelKey, AuthenticationKey


def get_relay_instance(relay_type: str):
    if relay_type == RelayType.openai:
        return OpenaiRelay()
    elif relay_type == RelayType.ali:
        return AliRelay()
    elif relay_type == RelayType.baichuan:
        return BaichuanRelay()
    elif relay_type == RelayType.baidu:
        return BaiduRelay()
    elif relay_type == RelayType.xunfei:
        return XunfeiRelay()
    elif relay_type == RelayType.zhipu:
        return ZhipuRelay()
    elif relay_type == RelayType.moonshot:
        return MoonshotRelay()
    elif relay_type == RelayType.anthropic:
        return AnthropicRelay()
    elif relay_type == RelayType.gemini:
        return GeminiRelay()
    elif relay_type == RelayType.huawei:
        return HuaweiRelay()
    elif relay_type == RelayType.bytedance:
        return ByteDanceRelay()
    elif relay_type == RelayType.ai360:
        return AI360Relay()
    else:
        raise NotImplementedError


def main_distribute(auth_key: AuthenticationKey, chat_body: APIBody, key: ChannelKey):
    key_type = key.key_type
    relay_instance = get_relay_instance(key_type)
    return relay_instance.main(auth_key, chat_body, key)
