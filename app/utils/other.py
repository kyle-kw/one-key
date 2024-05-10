# -*- coding: utf-8 -*-

# @Time    : 2024/4/18 13:26
# @Author  : kewei

import httpx
import orjson
import random
from typing import List
from datetime import datetime

from app.schemas.other import RerankBody, RerankResponse, RerankData, Usage, CalTokenResponse, CalTokenRes
from app.relay.request import _request
from app.utils.common import count_text_tokens
from app.utils.crud import get_channel_keys
from app.utils.crud.main import calculate_log_tokens
from app.utils.crud.models import ChannelKey, AuthenticationKey
from app.utils.exceptions import AuthException


async def rerank_relay_type_1(rerank_body: RerankBody, url: str) -> RerankResponse:
    json_data = rerank_body.dict()
    del json_data["model"]
    request = httpx.Request(
        "POST",
        url,
        headers={
            "Content-Type": "application/json",
        },
        json=json_data,
    )

    async for content in _request(request):
        content = orjson.loads(content)
        data = []
        for item in content:
            data.append(RerankData(
                score=item["score"],
                index=item["index"],
                text=rerank_body.texts[item["index"]]
            ))
        data.sort(key=lambda x: x.index)

        all_text = rerank_body.query + "".join(rerank_body.texts)
        prompt_tokens = count_text_tokens(all_text)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens
        )
        response = RerankResponse(
            data=data,
            query=rerank_body.query,
            usage=usage
        )

        return response


async def rerank_relay_type_2(rerank_body: RerankBody, url: str) -> RerankResponse:
    request = httpx.Request(
        "POST",
        url,
        headers={
            "Content-Type": "application/json",
        },
        json=rerank_body.dict(),
    )

    async for content in _request(request):
        return RerankResponse.parse_raw(content)


async def rerank_relay(rerank_body: RerankBody, auth_key: AuthenticationKey) -> RerankResponse:
    """
    转发rerank请求
    :param rerank_body:
    :param auth_key:
    :return:
    """
    channel_keys: List[ChannelKey] = get_channel_keys(key_status=1,
                                                      key_group=auth_key.key_group,
                                                      model_type="rerank",
                                                      api_model=rerank_body.model)

    if not channel_keys:
        raise AuthException(detail="No rerank channel key available.")

    one_key: ChannelKey = random.choices(channel_keys,
                                         weights=[k.api_weight for k in channel_keys],
                                         k=1)[0]
    api_config = one_key.api_config or {}
    if api_config.get("type") == 'type1':
        return await rerank_relay_type_1(rerank_body, one_key.api_base)
    else:
        return await rerank_relay_type_2(rerank_body, one_key.api_base)


def calculate_tokens(api_key_pk: int,
                     start_time: datetime,
                     end_time: datetime = None) -> CalTokenResponse:
    res = calculate_log_tokens(api_key_pk, start_time, end_time)

    response = CalTokenResponse(
        message='查询成功',
        data=CalTokenRes(
            all_prompt_token=res['all_prompt_token'],
            all_completion_token=res['all_completion_token'],
            all_token=res['all_token'],
        )
    )

    return response
