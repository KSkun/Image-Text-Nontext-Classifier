from abc import ABC, abstractmethod
from typing import List

import src.net.net


class ClassifierCmd(ABC):
    # constants
    redis_id: str

    def __init__(self, redis_id: str):
        self.redis_id = redis_id

    @abstractmethod
    def execute(self) -> bool:
        pass


class ClassifyOneCmd(ClassifierCmd):
    """Classify one image command"""

    # constants
    image_file: str

    def __init__(self, redis_id: str, image_file: str):
        super().__init__(redis_id)
        self.image_file = image_file

    def execute(self) -> bool:
        net = src.net.net.net
        return net.predict(self.image_file)


class ClassifyManyCmd(ClassifierCmd):
    """Classify many image command"""

    # constants
    image_files: List[str]

    def __init__(self, redis_id: str, image_files: List[str]):
        super().__init__(redis_id)
        self.image_files = image_files

    def execute(self) -> bool:
        net = src.net.net.net
        return net.predict_many(self.image_files)
