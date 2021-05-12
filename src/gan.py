import torch
from torch import nn, optim

class GANLoss(nn.Module):
  def __init__(self, real_label=1., fake_label=0.):
    super().__init__()

    self.register_buffer("real_label", torch.tensor(real_label))
    self.register_buffer("fake_label", torch.tensor(fake_label))

    self.loss = nn.BCEWithLogitsLoss()

  def get_labels(self, preds, target_is_real):
    if target_is_real:
      return self.real_label.expand_as(preds)

    return self.fake_label.expand_as(preds)

  def __call__(self, preds, target_is_real):
    labels = self.get_labels(preds, target_is_real)

    return self.loss(preds, labels)

class MainModel(nn.Module):
  def __init__(self, device, D, G, lr_G=2e-4, lr_D=2e-4, beta1=.5, beta2=.999, lambda_L1=100.):
    super().__init__()

    self.device = device

    self.D = D
    self.G = G

    self.GANcriterion = GANLoss().to(self.device)
    self.L1criterion = nn.L1Loss()

    self.opt_G = optim.Adam(self.G.parameters(), lr=lr_G, betas=(beta1, beta2))
    self.opt_D = optim.Adam(self.D.parameters(), lr=lr_D, betas=(beta1, beta2))

    self.lambda_L1 = lambda_L1

  def set_requires_grad(self, model, requires_grad):
    for parameter in model.parameters():
      parameter.requires_grad = requires_grad

  def setup_input(self, data):
    self.L = data["L"].to(self.device)
    self.ab = data["ab"].to(self.device)

  def forward(self):
    self.fake_color = self.G(self.L).to(device)

  def backward_D(self):
    fake_image = torch.cat([self.L, self.fake_color], dim=1)
    fake_preds = self.D(fake_image.detach())

    self.loss_D_fake = self.GANcriterion(fake_preds, False)

    real_image = torch.cat([self.L, self.ab], dim=1)
    real_preds = self.D(real_image)

    self.loss_D_real = self.GANcriterion(real_preds, True)

    self.loss_D = (self.loss_D_fake + self.loss_D_real) * .5

    self.loss_D.backward()

  def backward_G(self):
    fake_image = torch.cat([self.L, self.fake_color], dim=1)
    fake_preds = self.D(fake_image)

    self.loss_G_GAN = self.GANcriterion(fake_preds, True)
    self.loss_G_L1 = self.lambda_L1 * self.L1criterion(self.fake_color, self.ab)

    self.loss_G = self.loss_G_GAN + self.loss_G_L1

    self.loss_G.backward()

  def optimize(self):
    self.forward()

    self.D.train()
    self.set_requires_grad(self.D, True)
    self.opt_D.zero_grad()
    self.backward_D()
    self.opt_D.step()

    self.G.train()
    self.set_requires_grad(self.D, False)
    self.opt_G.zero_grad()
    self.backward_G()
    self.opt_G.step()