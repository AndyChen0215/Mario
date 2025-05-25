import copy

from torch import nn


class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim, device=None):
        super().__init__()
        c, h, w = input_dim
        self.device = device if device is not None else ("cuda" if nn.Module().cuda().is_available() else "cpu")
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        ).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        input = input.to(self.device)
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
