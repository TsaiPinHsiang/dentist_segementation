import torch
from tqdm import tqdm

def train(net, trainLoader, optimizer, device, epochs):
    net.train()
    for epoch in range(epochs):
