"""
Содержит класс с TinyVGG моделью
"""
import torch
from torch import nn


class TinyVGG(nn.Module):
  """
  TinyVGG модель для мультиклассовой классификации изображений 64х64

  Args:
    in_features: Целое число входных каналов.
    hidden_features: Целое число скрытых юнитов.
    out_features: Целое число выходных каналов (равно количеству классов).
  """
  def __init__(
      self,
      in_features: int,
      hidden_features: int,
      out_features: int
    ):
    super().__init__()
    
    self.layer1 = nn.Sequential(
        nn.Conv2d(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=3,
            stride=1,
            padding=0
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=hidden_features,
            out_channels=hidden_features,
            kernel_size=3,
            stride=1,
            padding=0
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.layer2 = nn.Sequential(
        nn.Conv2d(
            in_channels=hidden_features,
            out_channels=hidden_features,
            kernel_size=3,
            stride=1,
            padding=0
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=hidden_features,
            out_channels=hidden_features,
            kernel_size=3,
            stride=1,
            padding=0
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            in_features=hidden_features*13*13, # для 64х64
            out_features=out_features
        )
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classifier(self.layer2(self.layer1(x)))
