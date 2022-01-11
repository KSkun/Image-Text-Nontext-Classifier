import os
from abc import ABC, abstractmethod
from shutil import copy
from typing import List, Union, Dict

from net.net import net
from config import C
from db.image import update_image_class


class ClassifierCmd(ABC):
    """Abstract class for classifier commands"""

    # constants
    redis_id: str  # redis stream object id
    task_id: str  # task ObjectId in mongodb

    def __init__(self, redis_id: str, task_id: str):
        self.redis_id = redis_id
        self.task_id = task_id

    @abstractmethod
    def execute(self) -> Union[bool, List[bool]]:
        """Implement the execution logic in child classes"""
        pass


class ClassifyInitCmd(ClassifierCmd):
    """Initialize command"""

    def __init__(self, redis_id: str):
        super().__init__(redis_id, '')
        return

    def execute(self) -> bool:
        # do nothing
        return False


class ClassifyOneCmd(ClassifierCmd):
    """Classify one image command"""

    # constants
    image_file: Dict[str, str]

    def __init__(self, redis_id: str, task_id: str, image_file: Dict[str, str]):
        super().__init__(redis_id, task_id)
        self.image_file = image_file

    def execute(self) -> bool:
        # do predict
        file_path = C.image_tmp_dir + '/' + self.task_id + '/' + self.image_file['file']
        result = net.predict(file_path)
        klass = 'text'
        if not result:
            klass = 'nontext'

        # update mongodb
        update_image_class(self.image_file['id'], klass)

        # copy image to classified dir
        os.makedirs(C.image_text_dir + '/' + self.task_id, exist_ok=True)
        os.makedirs(C.image_nontext_dir + '/' + self.task_id, exist_ok=True)
        if result:
            copy(file_path, C.image_text_dir + '/' + self.task_id + '/' + self.image_file['file'])
        else:
            copy(file_path, C.image_nontext_dir + '/' + self.task_id + '/' + self.image_file['file'])

        return result


class ClassifyManyCmd(ClassifierCmd):
    """Classify many image command"""

    # constants
    image_files: List[Dict[str, str]]

    def __init__(self, redis_id: str, task_id: str, image_files: List[Dict[str, str]]):
        super().__init__(redis_id, task_id)
        self.image_files = image_files

    def execute(self) -> List[bool]:
        # filename to file path
        images = []
        for i in self.image_files:
            file_path = C.image_tmp_dir + '/' + self.task_id + '/' + i['file']
            images.append(file_path)

        # do predict
        results = net.predict_many(images)

        os.makedirs(C.image_text_dir + '/' + self.task_id, exist_ok=True)
        os.makedirs(C.image_nontext_dir + '/' + self.task_id, exist_ok=True)
        for i in range(len(results)):
            klass = 'text'
            if not results[i]:
                klass = 'nontext'

            # update mongodb
            update_image_class(self.image_files[i]['id'], klass)

            # copy image to classified dir
            file_path = C.image_tmp_dir + '/' + self.task_id + '/' + self.image_files[i]['file']
            if results[i]:
                copy(file_path, C.image_text_dir + '/' + self.task_id + '/' + self.image_files[i]['file'])
            else:
                copy(file_path, C.image_nontext_dir + '/' + self.task_id + '/' + self.image_files[i]['file'])

        return results
