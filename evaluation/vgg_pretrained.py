from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.optim as optim


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class VGG_pretrained():
    def __init__(self, device=None):
        if device is None:
            device = 'cpu'
        self.input_size = 224
        self.num_classes = None
        self.num_epochs = 15
        self.device = device
        self.feature_extract = True

    def __str__(self):
        return "VGG pretraind, %d epochs fine tuning, imagesize: %d" % (self.num_epochs, self.input_size)


    @property
    def train_transforms(self):
        return [
            transforms.RandomResizedCrop(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    @property
    def inference_transforms(self):
        return [
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    def train(self, dataloader, log_dir):
        if self.num_classes is not None:
            assert self.num_classes == len(dataloader.dataset.classes)
        else:
            self.num_classes = len(dataloader.dataset.classes)

        model = models.vgg16(pretrained=True)
        set_parameter_requires_grad(model, self.feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
        model = model.to(self.device)
        params_to_update = model.parameters()
        if self.feature_extract:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model.train()
        for epoch in range(self.num_epochs):
            print("%s training epoch: %d" % (str(self), epoch))
            for idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()
        return model