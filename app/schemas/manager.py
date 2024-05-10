# -*- coding: utf-8 -*-

# @Time    : 2024/4/25 13:09
# @Author  : kewei

from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime, timedelta


class ChannelKeyPydantic(BaseModel):
    key_type: str = Field(...)
    key_group: int = 0
    key_status: int = 0
    api_key: str = Field(...)
    api_base: str = ''
    api_model: str = Field(...)
    model_type: str = 'chat'
    api_config: Optional[dict] = None
    api_weight: int = 1
    limit_token: int = 100
    comment: str = ''


class AuthenticationKeyPydantic(BaseModel):
    key_group: int = 0
    key_status: int = 0
    balance: int = -1
    api_key: str = Field(...)
    limit_token: int = 100
    model_mapping: Optional[dict] = None
    allow_models: str = ''
    expire_time: datetime = datetime.strptime('2999-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    comment: str = ''


class GlobalMappingPydantic(BaseModel):
    old_model: str
    new_model: str
    status: int = 0
    comment: str = ''


class OperateState(BaseModel):
    status: int = 200
    message: str = ''
    data: Any = None

