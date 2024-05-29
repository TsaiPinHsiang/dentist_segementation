import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

from den_Dataset import denDataset
from maskrcnn import get_instance_segmentation_model

if __name__ == '__main__':

    # Parser initializing
    print("Parser initializing")
    parser = argparse.ArgumentParser(description="Orchid classification")
    parser.add_argument('--ngpu', default=0, type=int, required=False)
    parser.add_argument('--img_size', default=1024, type=int, required=False)
    parser.add_argument('--root', default="dataset", type=str, required=False)
    parser.add_argument('--label_path', default="NTNU_mask", type=str, required=False)
    parser.add_argument('--img_path', default="Sample", type=str, required=False)
    parser.add_argument('--batch_size', default=2, type=int, required=False)
    parser.add_argument('--epochs', default=10, type=int, required=False)
    parser.add_argument('--lr_rate', default=0.001, type=float, required=False)
    args = parser.parse_args()

    numClasses = 2

    # Device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

    # define transforms
    dataTransformsTrain = transforms.Compose([
        transforms.Resize(size=args.img_size, max_size=args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load dataset
    print('loading dataset')
    trainSet = denDataset(args.root, args.img_path, args.label_path, transforms=dataTransformsTrain)
    data_loader = torch.utils.data.DataLoader(trainSet, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print('loading complete')

    # load maskrcnn model
    print('load model')
    model = get_instance_segmentation_model(numClasses)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, args.lr_rate, momentum=0.9, weight_decay=0.0005)

    # the learning rate scheduler decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


