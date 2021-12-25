import gc
import os
from math import ceil
from typing import Any, List

import torch
from PIL import Image
from torchvision import transforms

_model_file = '20211224-3-epoch26.pth'
_batch_size = 8


class ClassifierNet:
    """PyTorch net for text/non-text classifier"""

    __device: str
    __model: torch.nn.Module
    __transformation: Any

    def __init__(self):
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"

        self_dir = os.path.split(os.path.abspath(__file__))[0]
        self.__model = torch.load(self_dir + '/' + _model_file).to(self.__device)
        self.__model.eval()

        self.__transformation = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def predict(self, img_file: str) -> bool:
        gc.collect()
        with torch.no_grad():
            img = Image.open(img_file).convert('RGB')
            img = self.__transformation(img)
            img = img.unsqueeze(0)
            img = img.to(self.__device)
            out = self.__model(img)
            pred = torch.max(out, 1)[1].item()
        return pred == 1

    def predict_many(self, image_files: List[str]) -> List[bool]:
        gc.collect()
        pred: List[bool] = []
        with torch.no_grad():
            iMax = ceil(len(image_files) / _batch_size)
            for i in range(iMax):
                batch = image_files[i * _batch_size: min((i + 1) * _batch_size, len(image_files))]
                batch_tensors = []
                for img_file in batch:
                    img = Image.open(img_file).convert('RGB')
                    img = self.__transformation(img)
                    batch_tensors.append(img.to(self.__device))
                batch_tensor = torch.stack(batch_tensors)
                out = self.__model(batch_tensor)
                batch_pred = torch.max(out, 1)[1]
                for res in batch_pred:
                    pred.append(res.item() == 1)
        return pred


net = ClassifierNet()
