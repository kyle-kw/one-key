# -*- coding: utf-8 -*-

# @Time    : 2024/4/18 13:22
# @Author  : kewei

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class RerankBody(BaseModel):
    model: str
    query: str
    texts: List[str]


class RerankData(BaseModel):
    object: str = Field(description="The object type.", default="embedding")
    score: float
    index: int
    text: str


class Usage(BaseModel):
    prompt_tokens: int = Field(description="The number of tokens in the prompt.", default=0)
    total_tokens: int = Field(description="The total number of tokens in the prompt and response.", default=0)


class RerankResponse(BaseModel):
    object: str = Field(description="The object type.", default="list")
    data: List[RerankData]
    query: str = Field(default="")
    model: str = Field(default="rerank-ada-002")
    usage: Optional[Usage] = None


class CalTokenRes(BaseModel):
    all_prompt_token: int
    all_completion_token: int
    all_token: int


class CalTokenResponse(BaseModel):
    status: int = 200
    message: str = ''
    data: CalTokenRes = None
