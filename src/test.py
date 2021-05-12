import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.resnet import resnet18

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  G = build_generator(1, 2, resnet18).to(device)
  G.load_state_dict(torch.load("final_model.pt", map_location=device))

  img = Image.open("input_image_name")

  t = 2*(transforms.ToTensor()(img))[0] - 1
  t = torch.unsqueeze(torch.unsqueeze(t, 0), 0).to(device)


  output = G(t)

  out_img = lab_to_rgb(t, output.detach())

  plt.axis("off")
  plt.imshow(out_img[0])