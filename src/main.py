import glob
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from fastai.data.external import untar_data, URLs

SIZE = 256

if __name__ == "__main__":
  root = str(untar_data(URLs.COCO_SAMPLE)) + "/train_sample"

  paths = glob.glob(root + "/*.jpg")

  np.random.seed(42)
  paths_subset = np.random.choice(paths, 12_000, replace=False)

  rand_idxs = np.random.permutation(12_000)
  train_idxs = rand_idxs[:10_000]
  val_idxs = rand_idxs[10_000:]

  train_paths = paths_subset[train_idxs]
  val_paths = paths_subset[val_idxs]

  train_dset = TrainingDataset(train_paths)
  val_dset = ValidationDataset(val_paths)

  train_dl = DataLoader(train_dset, 16, num_workers=2, pin_memory=True)
  val_dl = DataLoader(val_dset, 16, num_workers=2, pin_memory=True)


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  D = PatchDiscriminator(1, 2).to(device)
  G = build_generator(1, 2, resnet18).to(device)

  # l1_opt = optim.Adam(G.parameters(), lr=1e-4)
  # l1_loss = nn.L1Loss()

  # train_generator(device, G, train_dl, val_dl, l1_opt, l1_loss, 20)

  G.load_state_dict(torch.load("res18-unet.pt", map_location=device))


  model = MainModel(device, D, G)

  train_model(model, train_dl, val_dl, 20)