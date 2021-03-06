from bson import ObjectId

from db.main import mongo_db


def mark_task_done(task_id: str):
    """
    Mark classifier done in MongoDB

    :arg task_id: task ObjectId
    """
    mongo_db()['task'].update_one({'_id': ObjectId(task_id)}, {'$set': {'classifier_done': True}})
