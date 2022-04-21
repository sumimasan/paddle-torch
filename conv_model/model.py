import torch
from torch import nn
import paddle


"""initial model"""
class conv_torch_model(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3),
            torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3),
            torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3),
        )
        # torch.save(conv_torch_model.state_dict(), "../Data/torch.params")

    def forward(self,x):
        return self.net(x)


class conv_paddle_model:
    def __init__(self):
        self.net = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=3, out_channels=1, kernel_size=3),
            paddle.nn.Conv2D(in_channels=1, out_channels=1, kernel_size=3),
            paddle.nn.Conv2D(in_channels=1, out_channels=1, kernel_size=3),
        )


