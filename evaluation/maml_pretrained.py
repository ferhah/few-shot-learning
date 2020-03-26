import torch
from maml.model import ModelConvMiniImagenet
import maml.utils
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

    def train(self, dataloader, log_dir=None):
        self.log_dir = log_dir
        params = None
        for step in range(self.num_adaptation_steps):
            for imgs, labels in dataloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = self.model(imgs, params=params)
                inner_loss = self.loss_function(logits, labels)
                self.model.zero_grad()
                params = maml.utils.update_parameters(self.model, inner_loss,
                                                      params=None,
                                                      step_size=0.1, first_order=True)
                # TODO: Multiple batches did not work (yet). Therefore training batch size needs to be big enough to cover all training samples
                break
        return self, None
