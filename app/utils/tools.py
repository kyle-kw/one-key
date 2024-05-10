# -*- coding: utf-8 -*-

# @Time    : 2024/3/20 14:55
# @Author  : kewei

import random
from typing import AsyncIterator, List
from loguru import logger
from fastapi.responses import StreamingResponse, Response
from datetime import datetime

from app.utils.crud import get_auth_keys, get_channel_keys, get_global_mapping
from app.utils.crud.models import ChannelKey, AuthenticationKey
from app.utils.exceptions import AuthException, TokenLimitException, InternalException
from app.schemas.relay import ChatBody, ChatCompletion, ModelResponse, EmbeddingResponse
from app.relay.common import judge_model_type, get_type_model_list
from app.utils.common import auth_limit_key_token
from app.config import env_settings


async def auth_main(key: str) -> AuthenticationKey:
    auth_db_keys = get_auth_keys(api_key=key, key_status=1)
    if not auth_db_keys:
        logger.error('验证key失败！')
        raise AuthException(detail='验证key失败！')

    auth_db_key = auth_db_keys[0]

    if auth_db_key.expire_time < datetime.now():
        raise AuthException(detail='key已过期！')

    await auth_limit_key_token(auth_db_key.api_key, auth_db_key.limit_token * 1000)
    return auth_db_key


def auth_super_main(key: str) -> AuthenticationKey:
    auth_db_keys = get_auth_keys(api_key=key, key_status=1)
    if not auth_db_keys:
        logger.error('验证key失败！')
        raise AuthException(detail='验证key失败！')

    auth_db_key = auth_db_keys[0]

    if auth_db_key.expire_time < datetime.now():
        raise AuthException(detail='key已过期！')

    if auth_db_key.key_group < 100:
        raise AuthException(detail='非超级用户权限！')

    return auth_db_key


async def choice_not_limit_key(channel_keys: List[ChannelKey]) -> ChannelKey:
    while channel_keys:
        try:
            one_key = random.choices(channel_keys,
                                     weights=[k.api_weight for k in channel_keys],
                                     k=1)[0]
            await auth_limit_key_token(one_key.api_key, one_key.limit_token * 1000)
            return one_key
        except TokenLimitException:
            channel_keys.remove(one_key)
            logger.error(f'key: {one_key.api_key} 限流！')

    raise TokenLimitException(detail='channel key限流！')


async def get_one_channel_key(model_name: str, auth_key: AuthenticationKey) -> ChannelKey:
    if auth_key.allow_models and model_name not in auth_key.allow_models.split(','):
        logger.error(f'使用{model_name}不在允许的model列表中！')
        raise AuthException(detail=f'使用{model_name}不在允许的model列表中！')

    model_mapping = auth_key.model_mapping or {}
    key_group = auth_key.key_group

    new_model = model_mapping.get(model_name)
    if not new_model:
        global_mapping = get_global_mapping(key_status=1)
        for one in global_mapping:
            if one.old_model == model_name:
                new_model = one.new_model
                break

    if new_model:
        model_name = new_model

    if 'embedding' in model_name.lower():
        model_type = 'embedding'
    else:
        model_type = 'chat'

    allow_keys = get_channel_keys(key_group=key_group,
                                  api_model=model_name,
                                  key_status=1,
                                  model_type=model_type)
    if not allow_keys:
        api_type = judge_model_type(model_name)
        allow_keys = get_channel_keys(key_group=key_group,
                                      key_type=api_type,
                                      key_status=1,
                                      api_model='*')

    if not allow_keys:
        logger.error(f'没有可用的channel key！ model: {model_name}')
        raise AuthException(detail='没有可用的channel key！')

    if env_settings.limit_token:
        return await choice_not_limit_key(allow_keys)

    one_key = random.choices(allow_keys,
                             weights=[k.api_weight for k in allow_keys],
                             k=1)[0]

    return one_key


def get_allow_channel_keys_model(auth_key: AuthenticationKey) -> ModelResponse:
    if auth_key.allow_models:
        model_lst = []
        for one_model in auth_key.allow_models.split(','):
            if not one_model:
                continue

            model_lst.append({
                'id': one_model,
                'created': 0,
                'object': 'model',
                'owned_by': 'object',
            })
        return ModelResponse(data=model_lst)

    key_group = auth_key.key_group

    allow_keys = get_channel_keys(key_group=key_group,
                                  key_status=1)
    model_lst = []
    model_set = set()
    for one in allow_keys:
        if one.api_model == '*':
            models = get_type_model_list(one.key_type)
            for model in models:
                if model in model_set:
                    continue
                model_set.add(model)
                model_lst.append({
                    'id': model,
                    'created': int(one.create_time.timestamp()),
                    'object': 'model',
                    'owned_by': one.key_type,
                })
        else:
            if one.api_model in model_set:
                continue
            model_set.add(one.api_model)
            model_lst.append({
                'id': one.api_model,
                'created': int(one.create_time.timestamp()),
                'object': 'model',
                'owned_by': one.key_type,
            })
    return ModelResponse(data=model_lst)


async def convert_stream_response(first_value, r: AsyncIterator):
    yield 'data: ' + first_value.json() + '\n\n'

    async for chunk in r:
        yield 'data: ' + chunk.json() + '\n\n'
    yield 'data: [DONE]\n\n'


async def build_chat_response(chat_body: ChatBody, r: AsyncIterator):
    if chat_body.stream:
        try:
            first_value = await r.__anext__()
        except StopAsyncIteration:
            raise InternalException(detail='转发请求异常！')

        return StreamingResponse(
            convert_stream_response(first_value, r),
            status_code=200,
            media_type='text/event-stream',
        )
    else:
        all_content = None
        async for chunk in r:
            chunk: ChatCompletion
            if not all_content:
                all_content = chunk

        if all_content is None:
            raise InternalException(detail='转发请求异常！')

        return Response(content=all_content.json(),
                        status_code=200,
                        media_type='application/json')


async def build_embedding_response(r: AsyncIterator):
    all_content = None
    async for chunk in r:
        chunk: EmbeddingResponse
        if not all_content:
            all_content = chunk

    return Response(content=all_content.json(),
                    status_code=200,
                    media_type='application/json')
