# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, Request
from app.schemas.relay import ChatBody, EmbeddingBody, ModelResponse, EmbeddingResponse
from app.utils.logger import logger
from app.utils.tools import get_one_channel_key, get_allow_channel_keys_model, build_chat_response, \
    build_embedding_response
from app.dependencies.authentication import auth_depend
from app.utils.crud.models import AuthenticationKey
from app.relay import main_distribute

router = APIRouter(prefix="/v1")
compatible = APIRouter()


@router.get("/models")
async def models_relay(auth_key: AuthenticationKey = Depends(auth_depend)) -> ModelResponse:

    keys = get_allow_channel_keys_model(auth_key)

    return keys


@router.post("/chat/completions")
@compatible.post("/{prefix:path}/completions")
async def chat_completions_relay(chat_body: ChatBody,
                                 auth_key: AuthenticationKey = Depends(auth_depend)):
    logger.debug(f'chat: 开始转发请求服务。 {chat_body}')

    channel_key = await get_one_channel_key(chat_body.model, auth_key)
    r = main_distribute(auth_key, chat_body, channel_key)

    logger.debug('chat: 转发请求服务完成。')
    return await build_chat_response(chat_body, r)


@router.post("/embeddings")
async def embeddings_relay(embedding_body: EmbeddingBody,
                           auth_key: AuthenticationKey = Depends(auth_depend)) -> EmbeddingResponse:
    logger.debug(f'embedding: 开始转发请求服务。{embedding_body}')

    channel_key = await get_one_channel_key(embedding_body.model, auth_key)
    r = main_distribute(auth_key, embedding_body, channel_key)

    logger.debug('embedding: 转发请求服务完成。')
    return await build_embedding_response(r)


@router.post("/images/generations")
async def images_generations_relay(auth_key: AuthenticationKey = Depends(auth_depend)):
    raise NotImplementedError
