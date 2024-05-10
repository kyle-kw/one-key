# -*- coding: utf-8 -*-

# @Time    : 2024/4/24 17:21
# @Author  : kewei

import orjson
from typing import Union, List
from fastapi import APIRouter, Depends, Request

from app.utils.exceptions import RequestException
from app.dependencies.authentication import auth_super_depend
from app.utils.crud.models import AuthenticationKey, ChannelKey, GlobalMapping
from app.schemas.manager import ChannelKeyPydantic, AuthenticationKeyPydantic, GlobalMappingPydantic, OperateState
from app.utils.crud.main import insert_auth_key, insert_channel_key, insert_global_mapping, update_channel_key, \
    update_auth_key, update_global_mapping, delete_auth_key, delete_channel_key, delete_global_mapping

router = APIRouter(prefix='/v1')


@router.post('/add/channel-key')
def add_channel_key(channel_key: Union[ChannelKeyPydantic, List[ChannelKeyPydantic]],
                    auth_key: AuthenticationKey = Depends(auth_super_depend)):
    insert_data = []
    if isinstance(channel_key, list):
        for one_key in channel_key:
            insert_data.append(ChannelKey(**one_key.dict(exclude_defaults=True)))
    else:
        insert_data.append(ChannelKey(**channel_key.dict(exclude_defaults=True)))

    try:
        insert_channel_key(insert_data)
    except Exception as e:
        raise RequestException(detail=str(e))

    return OperateState(message='channel key插入成功。')


@router.post('/add/auth-key')
def add_auth_key(auth_key_dan: Union[AuthenticationKeyPydantic, List[AuthenticationKeyPydantic]],
                 auth_key: AuthenticationKey = Depends(auth_super_depend)):
    insert_data = []
    if isinstance(auth_key_dan, list):
        for one_key in auth_key_dan:
            insert_data.append(AuthenticationKey(**one_key.dict(exclude_defaults=True)))
    else:
        insert_data.append(AuthenticationKey(**auth_key_dan.dict(exclude_defaults=True)))

    try:
        insert_auth_key(insert_data)
    except Exception as e:
        raise RequestException(detail=str(e))

    return OperateState(message='authentication key插入成功。')


@router.post('/add/global-mapping')
def add_global_mapping(global_mapping: Union[GlobalMappingPydantic, List[GlobalMappingPydantic]],
                       auth_key: AuthenticationKey = Depends(auth_super_depend)):
    insert_data = []
    if isinstance(global_mapping, list):
        for one_key in global_mapping:
            insert_data.append(GlobalMapping(**one_key.dict(exclude_defaults=True)))
    else:
        insert_data.append(GlobalMapping(**global_mapping.dict(exclude_defaults=True)))

    try:
        insert_global_mapping(insert_data)
    except Exception as e:
        raise RequestException(detail=str(e))

    return OperateState(message='global mapping插入成功。')


@router.post('/update/channel-key')
async def update_channel_key_api(request: Request,
                                 auth_key: AuthenticationKey = Depends(auth_super_depend)):
    channel_key = await request.json()
    if 'api_key' not in channel_key or 'api_model' not in channel_key:
        raise RequestException(detail="body 缺少api_key or api_model.")

    try:
        update_channel_key(channel_key)
    except Exception as e:
        raise RequestException(detail=str(e))
    return OperateState(message='channel_key更新成功。')


@router.post('/update/auth-key')
async def update_auth_key_api(request: Request,
                              auth_key: AuthenticationKey = Depends(auth_super_depend)):
    auth_key_ = await request.json()
    if 'api_key' not in auth_key_:
        raise RequestException(detail="body 缺少api_key.")

    try:
        update_auth_key(auth_key_)
    except Exception as e:
        raise RequestException(detail=str(e))

    return OperateState(message='authentication key更新成功。')


@router.post('/update/global-mapping')
async def update_global_mapping_api(request: Request,
                                    auth_key: AuthenticationKey = Depends(auth_super_depend)):
    global_mapping = await request.json()
    if 'old_model' not in global_mapping or 'new_model' not in global_mapping:
        raise RequestException(detail="body 缺少api_key.")

    try:
        update_global_mapping(global_mapping)
    except Exception as e:
        raise RequestException(detail=str(e))

    return OperateState(message='global mapping更新成功。')


@router.post('/delete/channel-key')
async def delete_channel_key_api(request: Request,
                                 auth_key: AuthenticationKey = Depends(auth_super_depend)):
    channel_key = await request.json()
    if 'api_key' not in channel_key or 'api_model' not in channel_key:
        raise RequestException(detail="body 缺少api_key or api_model.")

    try:
        delete_channel_key(channel_key)
    except Exception as e:
        raise RequestException(detail=str(e))
    return OperateState(message='channel_key删除成功。')


@router.post('/delete/auth-key')
async def delete_auth_key_api(request: Request,
                              auth_key: AuthenticationKey = Depends(auth_super_depend)):
    auth_key_ = await request.json()
    if 'api_key' not in auth_key_:
        raise RequestException(detail="body 缺少api_key.")

    if auth_key_['api_key'] == auth_key.api_key:
        raise RequestException(detail="不能删除请求的key.")

    try:
        delete_auth_key(auth_key_)
    except Exception as e:
        raise RequestException(detail=str(e))

    return OperateState(message='authentication key删除成功。')


@router.post('/delete/global-mapping')
async def delete_global_mapping_api(request: Request,
                                    auth_key: AuthenticationKey = Depends(auth_super_depend)):
    global_mapping = await request.json()
    if 'old_model' not in global_mapping or 'new_model' not in global_mapping:
        raise RequestException(detail="body 缺少api_key.")

    try:
        delete_global_mapping(global_mapping)
    except Exception as e:
        raise RequestException(detail=str(e))

    return OperateState(message='global mapping删除成功。')

