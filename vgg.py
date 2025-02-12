import torch
from torch import nn
from collections import OrderedDict, defaultdict


class VGG(nn.Module):
    ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    # m = Max Pooling Layer

    def __init__(self) -> None:  # -> Specifies the return type of the function
        super().__init__()

        layers = []
        # A default-dict is a type of dictionary provided by the collections module in Python. It is
        # similar to a regular dictionary, but with a key difference: it provides a default value for
        # any key that does not exist in the dictionary, instead of raising a KeyError.
        #
        # The input parameter is known as a default factory.  The default_factory in a default-dict is
        # a callable (like a function) that provides default values for missing keys. When you try to
        # access a key that doesn't exist in the dictionary, the default_factory is called to generate
        # a default value for that key. Specifying int returns a zero. 0 cannot be specified directly
        # as it is not callable.
        #
        # This is frequently used to count occurrences of items in a list. See the Add function
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != 'M':
                # conv-bn-relu
                # 3 = kernel_size
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))  # True = Inplace, No xtra mem allocated. replaces input.
                in_channels = x
            else:
                # max pool
                add("pool", nn.MaxPool2d(2))

        # add an average pooling layer
        # Before average pooling [N, 3, 32, 32] --> [N, 512, 2, 2]
        # after average pooling [N, 512, 2, 2] --> [N, 512, 1, 1]
        add("avgpool", nn.AvgPool2d(2))

        # nn.Sequential can take an OrderedDict to create a sequence of layers with specific names.
        #  This makes it easier to refer to and debug individual layers. For example, if you want
        #  to access the first convolutional layer, you can refer to it by its name
        #  (e.g., model.backbone['conv0']) if the layers are named. If a list was used directly, we would
        # have to reference the layers by their index which is harder to debug.
        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone: [N, 3, 32, 32] => [N, 512, 1, 1]
        x = self.backbone(x)

        x = x.view(x.shape[0], -1)  # squeeze from 4D to 2D

        # classifier: [N, 512] => [N, 10]
        x = self.classifier(x)
        return x
