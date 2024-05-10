
from fastapi import HTTPException, status
from typing import Any, Optional, Dict


class AuthException(HTTPException):
    def __init__(
        self,
        detail: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        status_code = status.HTTP_401_UNAUTHORIZED
        detail = detail or '身份验证失败！'
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class RequestException(HTTPException):
    def __init__(
        self,
        detail: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        status_code = status.HTTP_400_BAD_REQUEST
        detail = detail or '无效请求！'
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class InternalException(HTTPException):
    def __init__(
        self,
        detail: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        detail = detail or '内部错误！'
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class TimeoutException(HTTPException):
    def __init__(
        self,
        detail: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        status_code = status.HTTP_408_REQUEST_TIMEOUT
        detail = detail or '请求超时！'
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class TokenLimitException(HTTPException):
    def __init__(
        self,
        detail: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        status_code = status.HTTP_429_TOO_MANY_REQUESTS
        detail = detail or 'token限流触发！'
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class OtherException(HTTPException):
    def __init__(
        self,
        detail: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        status_code = status.HTTP_400_BAD_REQUEST
        detail = detail or '请求失败！'
        super().__init__(status_code=status_code, detail=detail, headers=headers)


