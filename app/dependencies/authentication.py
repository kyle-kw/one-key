# -*- coding: utf-8 -*-

from fastapi import Request
from loguru import logger
from app.utils.exceptions import AuthException
from app.utils.tools import auth_main, auth_super_main
from app.utils.common import strip_prefix
from app.utils.crud.models import AuthenticationKey


async def auth_depend(request: Request) -> AuthenticationKey:
    key = request.headers.get('Authorization')
    if not key:
        logger.error('请求头缺少Authorization!')
        raise AuthException(detail='请求头缺少Authorization！')

    key = strip_prefix(key, 'Bearer ')

    auth_key = await auth_main(key)

    return auth_key


def auth_super_depend(request: Request) -> AuthenticationKey:
    key = request.headers.get('Authorization')
    if not key:
        logger.error('请求头缺少Authorization!')
        raise AuthException(detail='请求头缺少Authorization！')

    key = strip_prefix(key, 'Bearer ')

    auth_key = auth_super_main(key)

    return auth_key
