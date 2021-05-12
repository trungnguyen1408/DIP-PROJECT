from torch import nn

class CNNBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super().__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(
          in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
      ),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2),
    )

  def forward(self, x):
    return self.conv(x)

class PatchDiscriminator(nn.Module):
  def __init__(self, x_channels, y_channels, features=[64, 128, 256, 512]):
    super().__init__()

    self.initial = nn.Sequential(
      nn.Conv2d(x_channels + y_channels, features[0], 4, 2, 1, padding_mode="reflect"),
      nn.LeakyReLU(0.2),
    )

    layers = []
    for i in range(1, len(features)):
      layers.append(
        CNNBlock(features[i-1], features[i], 1 if features[i] == features[-1] else 2),
      )

    layers.append(
      nn.Conv2d(features[-1], 1, 4, 1, 1, padding_mode="reflect"),
    )

    self.model = nn.Sequential(*layers)

  def forward(self, xy):
    xy = self.initial(xy)

    return self.model(xy)