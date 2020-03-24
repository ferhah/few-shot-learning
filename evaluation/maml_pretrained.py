import torch
from maml.model import ModelConvMiniImagenet
from torchvision.transforms import ToTensor, Resize
import torch.nn
import torch.nn.functional as F

class MAML():
    def __init__(self, modelpath=None, device=None):
        if device is None:
            device = 'cpu'
        self.device = device
        self.modelpath = modelpath
        self.model = ModelConvMiniImagenet(10, hidden_size=64)
        if modelpath:
            self.model.load_state_dict(torch.load(self.modelpath))
        self.num_adaptation_steps=1
        self.loss_function = F.cross_entropy

    @property
    def train_transforms(self):
        return [Resize(84), ToTensor()]

    @property
    def inference_transforms(self):
        return [Resize(84), ToTensor()]