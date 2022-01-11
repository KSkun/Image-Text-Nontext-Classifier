import logging
from typing import Union

from pymongo import MongoClient
from pymongo.database import Database

from config import C

_mongo_client: Union[MongoClient, None] = None
_mongo_db: Union[Database, None] = None


def mongo_db() -> Database:
    global _mongo_db
    return _mongo_db


def init_db():
    """Init database objects"""
    global _mongo_client, _mongo_db

    _mongo_client = MongoClient(C.mongo_addr, C.mongo_port)
    try:
        _mongo_client.server_info()
    except Exception as e:
        logging.getLogger('image-classifier-backend').error('mongo error, ' + str(e))
        exit(1)
    _mongo_db = _mongo_client[C.mongo_db]
