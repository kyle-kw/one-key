# -*- coding: utf-8 -*-

# @Time    : 2024/4/15 16:46
# @Author  : kewei
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager


class CRUDSession:
    def __init__(self, url='sqlite:///test.db'):
        connect_args = {}
        if url.startswith('sqlite'):
            connect_args = {"check_same_thread": False}

        self.engine = create_engine(url, connect_args=connect_args)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=True)

    def init_database(self, base_obj):
        base_obj.metadata.create_all(self.engine)

    @contextmanager
    def get_session_new(self):
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
