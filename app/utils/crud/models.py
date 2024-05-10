# -*- coding: utf-8 -*-

from sqlalchemy import Column, DECIMAL, Index, Integer, JSON, String, Text, text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()
metadata = Base.metadata


class ChannelKey(Base):
    __tablename__ = '01_channel_keys'
    __table_args__ = (
        Index('idx_channel', 'api_key', 'api_model', unique=True),
    )

    record_pk = Column(Integer, primary_key=True, autoincrement=True)
    key_type = Column(String(100), nullable=False, comment='类型')
    key_group = Column(Integer, nullable=False, server_default=text("'0'"), comment='组权限。值越大，权限越大')
    key_status = Column(Integer, nullable=False, server_default=text("'0'"), comment='key状态, 1表示可用')
    api_key = Column(String(200), nullable=False, comment='key')
    api_base = Column(String(200), nullable=False, server_default=text("''"), comment='地址')
    api_model = Column(String(100), nullable=False, comment='模型')
    model_type = Column(String(100), nullable=False, server_default=text("'chat'"), comment='模型类型，chat、embedding、rerank')
    api_config = Column(JSON, comment='配置')
    api_weight = Column(Integer, nullable=False, server_default=text("'1'"), comment='权重')
    limit_token = Column(Integer, nullable=False, server_default=text("'100'"), comment='每分钟最大使用token数。单位：k')
    comment = Column(String(100), nullable=False, server_default=text("''"), comment='备注')
    create_time = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment='创建时间')
    update_time = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment='更新时间')


class AuthenticationKey(Base):
    __tablename__ = '02_authentication_keys'
    __table_args__ = (
        Index('idx_balance', 'api_key', 'balance'),
    )

    record_pk = Column(Integer, primary_key=True, autoincrement=True)
    key_group = Column(Integer, nullable=False, server_default=text("'0'"), comment='组权限。值越大，所需权限越大')
    key_status = Column(Integer, nullable=False, server_default=text("'0'"), comment='key状态')
    balance = Column(Integer, nullable=False, server_default=text("'-1'"), comment='余额')
    api_key = Column(String(200), nullable=False, unique=True, comment='key')
    limit_token = Column(Integer, nullable=False, server_default=text("'100'"), comment='每分钟最大使用token数。单位：k')
    model_mapping = Column(JSON, comment='配置')
    allow_models = Column(String(200), nullable=False, server_default=text("''"), comment='允许使用的model，使用,分割')
    expire_time = Column(DateTime, nullable=False, server_default=text("'2999-01-01 00:00:00'"), comment='过期时间')
    comment = Column(String(100), nullable=False, server_default=text("''"), comment='备注')
    create_time = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment='创建时间')
    update_time = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment='更新时间')


class GlobalMapping(Base):
    __tablename__ = '03_global_mapping'
    __table_args__ = (
        Index('idx_global', 'old_model', 'new_model', unique=True),
    )

    record_pk = Column(Integer, primary_key=True, autoincrement=True)
    old_model = Column(String(100), nullable=False, comment='源model')
    new_model = Column(String(100), nullable=False, comment='新model')
    status = Column(Integer, nullable=False, server_default=text("'0'"))
    comment = Column(String(100), nullable=False, server_default=text("''"), comment='备注')
    create_time = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment='创建时间')
    update_time = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment='更新时间')


class RecordLog(Base):
    __tablename__ = '04_record_logs'
    __table_args__ = (
        Index('idx_query', 'auth_key_pk', 'api_type', 'api_model', 'create_time'),
    )

    record_pk = Column(Integer, primary_key=True, autoincrement=True)
    auth_key_pk = Column(Integer, nullable=False, comment='验证使用key')
    api_type = Column(String(100), nullable=False, server_default=text("''"), comment='请求类型')
    user_id = Column(String(100), nullable=False, server_default=text("''"), comment='用户id')
    source_model = Column(String(100), nullable=False, server_default=text("''"), comment='原请求model名字')
    api_model = Column(String(100), nullable=False, server_default=text("''"), comment='请求model名字')
    prompt = Column(Text, comment='请求prompt')
    completion = Column(Text, comment='返回内容')
    prompt_token = Column(Integer, nullable=False, server_default=text("'0'"), comment='请求使用token数')
    completion_token = Column(Integer, nullable=False, server_default=text("'0'"), comment='响应使用token数')
    first_time = Column(DECIMAL(5, 2), nullable=False, server_default=text("'0.00'"), comment='请求第一次返回内容用时')
    all_time = Column(DECIMAL(5, 2), nullable=False, server_default=text("'0.00'"), comment='请求整体用时')
    return_rate = Column(DECIMAL(5, 2), nullable=False, server_default=text("'0.00'"), comment='吐字/s')
    api_body = Column(JSON, comment='请求body')
    retry_times = Column(Integer, nullable=False, server_default=text("'0'"), comment='重试次数')
    create_time = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment='创建时间')


