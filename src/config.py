import json


class ClassifierConfig:
    """
    Spider config data class
    See https://github.com/KSkun/Image-Text-Nontext-Classifier/blob/master/README.md
    """

    image_tmp_dir: str
    image_text_dir: str
    image_nontext_dir: str
    image_url: str

    mongo_addr: str
    mongo_port: int
    mongo_db: str

    redis_addr: str
    redis_port: int
    redis_db: int
    consumer_id: int


C = ClassifierConfig()


def load_config(file_path: str):
    """Load config fields to object C from file"""
    file = open(file_path, 'r')
    json_str = file.read()
    file.close()

    json_obj = json.loads(json_str)

    var_list = ClassifierConfig.__annotations__
    for v in var_list:
        if v in json_obj:
            setattr(C, v, json_obj[v])
        else:
            raise ValueError('missing field %s in config file %s' % (v, file_path))
