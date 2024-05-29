import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image


class denDataset(torch.utils.data.Dataset):
    def __init__(self, root, sdir, mdir, transforms=None):
        self.root = root
        self.sdir = sdir
        self.mdir = mdir
        self.transforms = transforms

        # load direction
        maskdir = list(sorted(os.listdir(os.path.join(root, mdir))))
        imgs = []
        masks = []
        for k in maskdir:
            tem_imgs = list(sorted(os.listdir(os.path.join(root, sdir, k))))
            tem_masks = list(sorted(os.listdir(os.path.join(root, mdir, k))))
            for i in range(len(tem_imgs)):
                tem_imgs[i] = k + '\\' + tem_imgs[i]
                tem_masks[i] = k + '\\' + tem_masks[i]

            imgs = imgs + tem_imgs
            masks = masks + tem_masks

        self.Imgs = imgs
        self.Masks = masks


    def __getitem__(self, idx):
        # load image path and mask path
        img_path = os.path.join(self.root, self.sdir, self.Imgs[idx])
        mask_path = os.path.join(self.root, self.mdir, self.Masks[idx])
        # load img
        img = Image.open(img_path).convert("RGB")
        # load mask
        # note that we haven't converted the mask to RGB
        mask = Image.open(mask_path)

        mask = np.array(mask)
        #instances are encoded as different colors
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.Imgs)
