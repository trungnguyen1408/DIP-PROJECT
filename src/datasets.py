import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from torchvision import transforms
from torch.utils.data import Dataset

class TrainingDataset(Dataset):
  def __init__(self, paths):
    self.transforms = transforms.Compose([
            transforms.Resize((SIZE, SIZE)),
            transforms.RandomHorizontalFlip(),
        ])
    self.paths = paths

  def __getitem__(self, idx):
    img = Image.open(self.paths[idx]).convert("RGB")
    img = self.transforms(img)
    img = np.array(img)

    lab_img = rgb2lab(img).astype("float32")
    lab_img = transforms.ToTensor()(lab_img)

    L = lab_img[[0], ...] / 50. - 1.
    ab = lab_img[[1, 2], ...] / 110.

    return {"L": L, "ab": ab}

  def __len__(self):
    return len(self.paths)

class ValidationDataset(Dataset):
  def __init__(self, paths):
    self.transforms = transforms.Compose([
            transforms.Resize((SIZE, SIZE)),
        ])
    self.paths = paths

  def __getitem__(self, idx):
    img = Image.open(self.paths[idx]).convert("RGB")
    img = self.transforms(img)
    img = np.array(img)

    lab_img = rgb2lab(img).astype("float32")
    lab_img = transforms.ToTensor()(lab_img)

    L = lab_img[[0], ...] / 50. - 1.
    ab = lab_img[[1, 2], ...] / 110.

    return {"L": L, "ab": ab}

  def __len__(self):
    return len(self.paths)