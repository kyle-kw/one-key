# -*- coding: utf-8 -*-

from cachetools import cached, TTLCache
from sqlalchemy import update, and_, delete
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError, ProgrammingError
from app.utils.crud.models import ChannelKey, AuthenticationKey, GlobalMapping, Base, RecordLog
from app.utils.crud.session import CRUDSession
from app.utils.common import generate_fd_key
from app.utils.logger import logger
from app.config import env_settings
from typing import List, Union
from datetime import datetime, timedelta

crud_session = CRUDSession(env_settings.db_url)


def _get_auth_keys(db: Session,
                   key_status: int = None,
                   key_group: int = None,
                   api_key: str = None):
    res = db.query(AuthenticationKey)
    if key_status is not None:
        res = res.filter(AuthenticationKey.key_status == key_status)
    if key_group is not None:
        res = res.filter(AuthenticationKey.key_group <= key_group)
    if api_key is not None:
        res = res.filter(AuthenticationKey.api_key == api_key)

    res = res.all()

    return res


def _get_channel_keys(db: Session,
                      key_status: int = None,
                      key_group: int = None,
                      api_model: str = None,
                      key_type: str = None,
                      model_type: str = None):
    res = db.query(ChannelKey)
    if key_status is not None:
        res = res.filter(ChannelKey.key_status == key_status)
    if key_group is not None:
        res = res.filter(ChannelKey.key_group <= key_group)
    if api_model is not None:
        res = res.filter(ChannelKey.api_model == api_model)
    if key_type is not None:
        res = res.filter(ChannelKey.key_type == key_type)
    if model_type is not None:
        res = res.filter(ChannelKey.model_type == model_type)

    res = res.all()

    return res


def _get_global_mapping(db: Session,
                        key_status: int = None):
    res = db.query(GlobalMapping)
    if key_status is not None:
        res = res.filter(GlobalMapping.status == key_status)
    res = res.all()

    return res


def init_database():
    with crud_session.get_session_new() as db:
        try:
            res = _get_global_mapping(db)
            logger.info('table exist')
        except (OperationalError, ProgrammingError) as e:
            if 'no such table' in str(e) or "doesn't exist" in str(e):
                crud_session.init_database(Base)
                fd_key = generate_fd_key()
                auth_key = AuthenticationKey(
                    key_group=100,
                    key_status=1,
                    api_key=fd_key
                )
                insert_auth_key(auth_key)
                logger.info(f'init table success. generate super user: {fd_key}')
            else:
                logger.error(e)


@cached(cache=TTLCache(maxsize=10, ttl=30))
def get_auth_keys(key_status: int = None,
                  key_group: int = None,
                  api_key: str = None) -> List[AuthenticationKey]:
    with crud_session.get_session_new() as db:
        return _get_auth_keys(db, key_status, key_group, api_key)


@cached(cache=TTLCache(maxsize=10, ttl=30))
def get_channel_keys(key_status: int = None,
                     key_group: int = None,
                     api_model: str = None,
                     key_type: str = None,
                     model_type: str = None) -> List[ChannelKey]:
    with crud_session.get_session_new() as db:
        return _get_channel_keys(db, key_status, key_group, api_model, key_type, model_type)


@cached(cache=TTLCache(maxsize=10, ttl=30))
def get_global_mapping(key_status: int = None) -> List[GlobalMapping]:
    with crud_session.get_session_new() as db:
        return _get_global_mapping(db, key_status)


def _insert_record_log(db: Session, record_log: Union[RecordLog, List[RecordLog]]):
    if isinstance(record_log, list):
        db.add_all(record_log)
    else:
        db.add(record_log)
    db.commit()


def insert_record_log(record_log: Union[RecordLog, List[RecordLog]]):
    with crud_session.get_session_new() as db:
        _insert_record_log(db, record_log)


def _insert_channel_key(db: Session, channel_key: Union[ChannelKey, List[ChannelKey]]):
    if isinstance(channel_key, list):
        db.add_all(channel_key)
    else:
        db.add(channel_key)
    db.commit()


def _insert_auth_key(db: Session, auth_key: Union[AuthenticationKey, List[AuthenticationKey]]):
    if isinstance(auth_key, list):
        db.add_all(auth_key)
    else:
        db.add(auth_key)
    db.commit()


def _insert_global_mapping(db: Session, global_mapping: Union[GlobalMapping, List[GlobalMapping]]):
    if isinstance(global_mapping, list):
        db.add_all(global_mapping)
    else:
        db.add(global_mapping)
    db.commit()


def insert_channel_key(channel_key: Union[ChannelKey, List[ChannelKey]]):
    with crud_session.get_session_new() as db:
        _insert_channel_key(db, channel_key)


def insert_auth_key(auth_key: Union[AuthenticationKey, List[AuthenticationKey]]):
    with crud_session.get_session_new() as db:
        _insert_auth_key(db, auth_key)


def insert_global_mapping(global_mapping: Union[GlobalMapping, List[GlobalMapping]]):
    with crud_session.get_session_new() as db:
        _insert_global_mapping(db, global_mapping)


def _update_channel_key(db: Session, channel_key: dict):
    api_key = channel_key.pop('api_key')
    api_model = channel_key.pop('api_model')
    update_sql = (
        update(ChannelKey).
        where(and_(ChannelKey.api_key == api_key,
                   ChannelKey.api_model == api_model)).
        values(**channel_key)
    )
    db.execute(update_sql)

    db.commit()


def update_channel_key(channel_key: dict):
    with crud_session.get_session_new() as db:
        _update_channel_key(db, channel_key)


def _update_auth_key(db: Session, auth_key: dict):
    api_key = auth_key.pop('api_key')
    update_sql = (
        update(AuthenticationKey).
        where(AuthenticationKey.api_key == api_key).
        values(**auth_key)
    )
    db.execute(update_sql)

    db.commit()


def update_auth_key(auth_key: dict):
    with crud_session.get_session_new() as db:
        _update_auth_key(db, auth_key)


def _update_global_mapping(db: Session, global_mapping: dict):
    old_model = global_mapping.pop('old_model')
    new_model = global_mapping.pop('new_model')
    update_sql = (
        update(GlobalMapping).
        where(and_(GlobalMapping.old_model == old_model,
                   GlobalMapping.new_model == new_model)).
        values(**global_mapping)
    )
    db.execute(update_sql)

    db.commit()


def update_global_mapping(global_mapping: dict):
    with crud_session.get_session_new() as db:
        _update_global_mapping(db, global_mapping)


def _delete_channel_key(db: Session, channel_key: dict):
    api_key = channel_key.pop('api_key')
    api_model = channel_key.pop('api_model')
    delete_sql = (
        delete(ChannelKey).
        where(and_(ChannelKey.api_key == api_key,
                   ChannelKey.api_model == api_model))
    )
    db.execute(delete_sql)

    db.commit()


def delete_channel_key(channel_key: dict):
    with crud_session.get_session_new() as db:
        _delete_channel_key(db, channel_key)


def _delete_auth_key(db: Session, auth_key: dict):
    api_key = auth_key.pop('api_key')
    delete_sql = (
        delete(AuthenticationKey).
        where(AuthenticationKey.api_key == api_key)
    )
    db.execute(delete_sql)

    db.commit()


def delete_auth_key(auth_key: dict):
    with crud_session.get_session_new() as db:
        _delete_auth_key(db, auth_key)


def _delete_global_mapping(db: Session, global_mapping: dict):
    old_model = global_mapping.pop('old_model')
    new_model = global_mapping.pop('new_model')
    delete_sql = (
        delete(GlobalMapping).
        where(and_(GlobalMapping.old_model == old_model,
                   GlobalMapping.new_model == new_model))
    )
    db.execute(delete_sql)

    db.commit()


def delete_global_mapping(global_mapping: dict):
    with crud_session.get_session_new() as db:
        _delete_global_mapping(db, global_mapping)


def _calculate_log_tokens(db: Session,
                          api_key_pk: int,
                          start_time: datetime,
                          end_time: datetime = None):
    res = db.query(RecordLog.prompt_token, RecordLog.completion_token).filter(RecordLog.auth_key_pk == api_key_pk)

    if end_time:
        res = res.filter(RecordLog.create_time >= start_time).filter(RecordLog.create_time <= end_time)
    else:
        end_time = start_time + timedelta(days=1)
        res = res.filter(RecordLog.create_time >= start_time).filter(RecordLog.create_time < end_time)

    res = res.all()

    all_prompt_token = 0
    all_completion_token = 0
    for one in res:
        one: RecordLog
        if one.prompt_token:
            all_prompt_token += one.prompt_token

        if one.completion_token:
            all_completion_token += one.completion_token

    all_token = all_prompt_token + all_completion_token

    response_data = {
        'all_prompt_token': all_prompt_token,
        'all_completion_token': all_completion_token,
        'all_token': all_token,
    }

    return response_data


@cached(cache=TTLCache(maxsize=10, ttl=30))
def calculate_log_tokens(api_key_pk: int,
                         start_time: datetime,
                         end_time: datetime = None):
    with crud_session.get_session_new() as db:
        return _calculate_log_tokens(db, api_key_pk, start_time, end_time)


if env_settings.init_database:
    init_database()

if __name__ == '__main__':
    init_database()
