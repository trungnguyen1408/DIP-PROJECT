class AverageMeter:
  def __init__(self):
    self.reset()

  def reset(self):
    self.count, self.avg, self.sum = 0., 0., 0.

  def update(self, val, count):
    self.count += count
    self.sum += val * count
    self.avg = self.sum / self.count

def init_loss_meters():
  loss_D_fake = AverageMeter()
  loss_D_real = AverageMeter()
  loss_D = AverageMeter()

  loss_G_GAN = AverageMeter()
  loss_G_L1 = AverageMeter()
  loss_G = AverageMeter()

  return {"loss_D_fake": loss_D_fake,
          "loss_D_real": loss_D_real,
          "loss_D": loss_D,
          "loss_G_GAN": loss_G_GAN,
          "loss_G_L1": loss_G_L1,
          "loss_G": loss_G}

def update_losses(model, loss_meters, count):
  for loss_name, loss_meter in loss_meters.items():
    loss = getattr(model, loss_name)
    loss_meter.update(loss.item(), count)

def log_results(loss_meters):
  for loss_name, loss_meter in loss_meters.items():
    print(f"{loss_name}: {loss_meter.avg:.4f}")