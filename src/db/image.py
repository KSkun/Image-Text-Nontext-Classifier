from bson import ObjectId

from db.main import mongo_db


def update_image_class(img_id: str, klass: str):
    """
    Update image class field in MongoDB

    :arg img_id: image ObjectId
    :arg klass: image class
    """
    mongo_db()['image'].update_one({'_id': ObjectId(img_id)}, {'$set': {'class': klass}})
