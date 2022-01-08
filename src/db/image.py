from bson import ObjectId

from db.main import mongo_db


def update_image_class(img_id: str, klass: str):
    mongo_db()['image'].update_one({'_id': ObjectId(img_id)}, {'$set': {'class': klass}})
