from models import PretrainModel
from data import ChexpertLoader

import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast

# from warmup_scheduler import GradualWarmupScheduler



def train_model(train_loader, model, criterion, optimizer, device):
  model.train()
  train_loss = []
  if device.type == 'cuda' or device.type == 'cpu':
    loop = tqdm(train_loader)
    for images, labels in loop:
      images = images.to(device)
      labels = labels.to(device)
      with torch.cuda.amp.autocast(): 
        outputs = model(images)
        loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())
      # loop.set_description('Epoch {:02d}/{:02d} | LR: {:.5f}'.format(epoch,
      #                                                                epochs-1,
      #                                                                   optimizer.param_groups[0]['lr']))
      loop.set_postfix(loss=np.mean(train_loss))
  else: #TODO TPU XLA code
    print("TPU")

  return train_loss # = np.mean(train_loss)


def eval_model(val_loader, model, criterion, device):
  model.eval()
  val_loss = 0.0
  if device.type == 'cuda' or device.type == 'cpu':
    for images, labels in tqdm(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(), torch.no_grad():
          outputs = model(images)
          loss = criterion(outputs.float(), labels)
                
        val_loss += loss.item() * images.size(0)
  return val_loss / len(val_loader.dataset)

train, val = ChexpertLoader()
vgg16_model = PretrainModel(backbone="vgg16", 
                            weights="imagenet",
                            num_classes=14,
                            linear_in_features=512*12*10,
                            )

for i in range(3):
  train_model(vgg16_model, )