import json


class ClassifierConfig:
    """Spider config data class"""

    redis_addr: str
    redis_port: int
    redis_db: int
    consumer_id: int


C = ClassifierConfig()


def load_config(file_path: str):
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
