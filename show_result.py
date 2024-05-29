import torch
import os
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from maskrcnn import get_instance_segmentation_model

def tensor2image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor.permute(1, 2, 0), dtype=np.uint8)
  return Image.fromarray(tensor)


if __name__ == "__main__":

    model = get_instance_segmentation_model(2)

    # load weight from ./ckpts/
    param_dict = torch.load("./ckpts/best_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(param_dict)
    model.eval()


  # Test 1 image
    # image_path = "IMG_7627.JPG"

  # Test all image
  # path setting
  # load every file in ./dataset/Sample
    root = 'dataset'
    samdir = 'Sample'

  # load direction
    imgdir = list(sorted(os.listdir(os.path.join(root, samdir))))
    imgs = []

    for k in imgdir:
      tem_imgs = list(sorted(os.listdir(os.path.join(root, samdir, k))))
      for i in range(len(tem_imgs)):
        tem_imgs[i] = k + '/' + tem_imgs[i]
      imgs = imgs + tem_imgs
    
  # bounding box threshold
    bb_thre = 0.8

  # ubuntu find font from path
  # fnt = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", size=50)


  # put every image in path and show result in corresponding folder
  # det means detection, seg means segmentation
    for index in tqdm(imgs):
      # print(index)
      image = Image.open(f"./{root}/{samdir}/{index}")
      test = index.split("/")
      check = test[-1].split('.')

      images = torch.as_tensor(np.array(image)[...,:3]/255, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
      output = model(images)[0]

      image = tensor2image(images[0])
      img = image.copy()
      black_img = Image.new('RGB', img.size, (0, 0, 0))
      draw = ImageDraw.Draw(img)
      scores = output["scores"]
      boxes = output["boxes"]

    # processing bounding box
      for box, score in zip(boxes, scores):
          if score.item() > bb_thre:
              box = [b.item() for b in box]
              x1, y1, x2 ,y2 = box
              x1 = int(x1)
              y1 = int(y1)
              x2 = int(x2)
              y2 = int(y2)
              color = tuple(np.random.choice(range(256), size=3))
              draw.line([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)], fill=color, width=5)
              # draw.text((x1,y1), f"{score.item():.4f}", font=fnt) # draw confidence

      if len(check[-1]) < 4 :
        img.save(f"./result/{test[0]}/det/det_{test[1][:-3]}png")
      else:
        img.save(f"./result/{test[0]}/det/det_{test[1][:-4]}png")


    # processing mask
      masks = output["masks"]
      img = np.array(image.copy(), dtype=np.uint8)
      black_img = np.array(black_img, dtype=np.uint8)
      for mask,score in zip(masks, scores):
          if score.item() > bb_thre:
              mask = mask.detach().squeeze().numpy()
              color = list(np.random.choice(range(256), size=3))
              
              img[np.where(mask>0.6)] = color
              black_img[np.where(mask>0.6)] = color
      img = Image.fromarray(img)
      black_img = Image.fromarray(black_img)
      if len(check[-1]) < 4 :
        img.save(f"./result/{test[0]}/seg/seg_{test[1][:-3]}png")
        black_img.save(f"./result/{test[0]}/seg/mask_{test[1][:-3]}png")
      else:
        img.save(f"./result/{test[0]}/seg/seg_{test[1][:-4]}png")

      # exit()