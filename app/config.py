# -*- coding: utf-8 -*-
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    db_url: str = 'mysql+pymysql://root:OVFEirCr3vmX49xU@localhost:3306/openai'
    redis_url: Optional[str] = 'redis://:zyWGcdW4QOMcjXpw@localhost:6379/0'

    limit_token: bool = False
    save_log: bool = False
    init_database: bool = False

    open_sentry: bool = False
    sentry_dsn: str = ''

    request_timeout: int = 10


env_settings = EnvSettings()
print(env_settings.dict())
