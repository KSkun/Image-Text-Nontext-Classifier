import gc
import os
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.autograd import Variable
from torchvision import transforms

_model_file = '20211224-3-epoch26.pth'


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

    def predict(self, img_file: str):
        gc.collect()
        with torch.no_grad():
            img = Image.open(img_file).convert('RGB')
            img = self.__transformation(img)
            img = img.unsqueeze(0)
            img = img.to(self.__device)
            out = self.__model(img)
            pred = torch.max(out, 1)[1].item()
        return pred == 1
