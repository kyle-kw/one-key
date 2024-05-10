from pydantic import BaseModel, Field
from typing import List, Optional, Union, Mapping, Any


class ChatBody(BaseModel):
    model: str = Field(...)
    messages: List[Mapping[str, str]] = Field(...)
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None

    frequency_penalty: Any = None
    logit_bias: Any = None
    logprobs: Any = None
    top_logprobs: Any = None
    response_format: Any = None
    seed: Any = None
    tools: Any = None
    tool_choice: Any = None
    function_call: Any = None
    functions: Any = None


class ChatChoice(BaseModel):
    index: int = Field(...)
    message: Mapping[str, Union[str, None]] = Field(...)
    logprobs: Any = None
    finish_reason: Optional[str] = None


class ChatCompletion(BaseModel):
    id: str = Field(...)
    object: str = Field(...)
    created: int = Field(...)
    model: str = Field(...)
    system_fingerprint: Optional[str] = None
    choices: List[ChatChoice] = Field(...)
    usage: Optional[Mapping[str, int]] = None


class StreamChatChoice(BaseModel):
    index: int = Field(...)
    delta: Mapping[str, Union[str, None]] = Field(...)
    logprobs: Any = None
    finish_reason: Optional[str] = None


class StreamChatCompletion(BaseModel):
    id: str = Field(...)
    object: str = Field(default='')
    created: int = Field(default=0)
    model: str = Field(...)
    system_fingerprint: Optional[str] = None
    choices: List[StreamChatChoice] = Field(...)


class StreamEndChatCompletion(BaseModel):
    id: str = Field(...)
    object: str = Field(default='')
    created: int = Field(default=0)
    model: str = Field(...)
    system_fingerprint: Optional[str] = None
    choices: List[StreamChatChoice] = Field(...)
    usage: Optional[Mapping[str, int]] = None


class EmbeddingBody(BaseModel):
    input: Union[List[Union[List[int], int, str]], str] = Field(...)
    model: str = Field(...)
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    index: int = Field(...)
    embedding: Union[List[float], str] = Field(...)
    object: str = Field(...)


class EmbeddingResponse(BaseModel):
    object: str = 'list'
    data: List[EmbeddingData] = Field(...)
    model: str = ''
    usage: Optional[Mapping[str, int]] = None


class ModelResponse(BaseModel):
    object: str = 'list'
    data: List[Mapping[str, Union[str, int]]] = Field(...)


APIBody = Union[ChatBody, EmbeddingBody]
APIResponse = Union[ChatCompletion, StreamChatCompletion, EmbeddingResponse, ModelResponse]
