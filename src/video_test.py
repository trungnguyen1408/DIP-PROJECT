import cv2
import torch
from PIL import Image
from torchvision.models.resnet import resnet18

'''
use command
ffmpeg -r 60 -i ./color_frame/%d_out.jpg -c:v libx264 -vf fps=30 -pix_fmt yuv420p out.mp4
'''

if __name__ == "__main__":
  source = cv2.VideoCapture('videoplayback.mp4')
  idx = 0
  ret = True

  while ret:
    ret, img = source.read()
    idx += 1

    t = 2*(transforms.ToTensor()(img))[0] - 1
    t = torch.unsqueeze(torch.unsqueeze(t, 0), 0).to(device)
    ab = G(t)
    rgb = (255*lab_to_rgb(t, ab.detach())[0]).astype(np.uint8)

    out_img = Image.fromarray(rgb, "RGB")
    out_img.save(f"./color_frame/{idx}_out.jpg")

    key = cv2.waitKey(1)
    if key == ord("q"):
      break

  cv2.destroyAllWindows()
  source.release()