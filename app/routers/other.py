# -*- coding: utf-8 -*-

# @Time    : 2024/4/18 13:20
# @Author  : kewei

from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends
from app.schemas.other import RerankBody, RerankResponse, CalTokenResponse
from app.dependencies.authentication import auth_depend
from app.utils.crud.models import AuthenticationKey
from app.utils.logger import logger
from app.utils.other import rerank_relay, calculate_tokens
from app.utils.exceptions import InternalException

router = APIRouter(prefix='/v1')


@router.post("/rerank")
async def rerank_relay_api(rerank_body: RerankBody,
                           auth_key: AuthenticationKey = Depends(auth_depend)) -> RerankResponse:
    logger.debug('rerank: 开始转发请求服务。')

    try:
        response = await rerank_relay(rerank_body, auth_key)
    except Exception as e:
        logger.exception(e)
        raise InternalException(detail=str(e))

    return response


@router.get("/tokens")
async def tokens_users(start_time: datetime,
                       end_time: Optional[datetime] = None,
                       auth_key: AuthenticationKey = Depends(auth_depend)) -> CalTokenResponse:
    logger.debug('tokens: 计算使用tokens。')
    try:
        response = calculate_tokens(auth_key.record_pk,
                                    start_time,
                                    end_time)
    except Exception as e:
        logger.exception(e)
        raise InternalException(detail=str(e))

    return response
