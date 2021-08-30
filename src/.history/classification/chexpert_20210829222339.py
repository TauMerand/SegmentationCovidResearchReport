from models import PretrainClassifier
from data import ChexpertLoader

import numpy as np
import time

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torch.cuda.amp import autocast

# from warmup_scheduler import GradualWarmupScheduler



def train_model(model, train_loader, criterion, optimizer, device, save_freq):
  model.train()
  train_loss = []
  if device.type == 'cuda' or device.type == 'cpu':
    loop = tqdm(train_loader)
    for i, images, labels in enumerate(loop):
      images = images.to(device)
      labels = labels.to(device)
      with torch.cuda.amp.autocast(): 
        outputs = model(images)
        loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())
      # if save_freq and (i % save_freq == 0):

      # loop.set_description('Epoch {:02d}/{:02d} | LR: {:.5f}'.format(epoch,
      #                                                                epochs-1,
      #                                                                   optimizer.param_groups[0]['lr']))
      loop.set_postfix(loss=np.mean(train_loss))
  else: #TODO TPU XLA code
    print("TPU")

  return train_loss # = np.mean(train_loss)


def eval_model(model, val_loader, criterion, device):
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
  else: #TODO TPU Code
    print('TPU')  
  return val_loss / len(val_loader.dataset)


def vgg16_classifier(epochs, device):
  start=time.time()
  train, val = ChexpertLoader()
  vgg16_classifier = PretrainClassifier(backbone="vgg16", 
                                        weights="imagenet",
                                        num_classes=14,
                                        linear_in_features=512*12*10,
                                        )
  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam( vgg16_classifier.parameters(),
                                lr=0.001)
  loop = tqdm(epochs)
  for i in loop:
    train_model(vgg16_classifier, train, criterion, optimizer, device)
    eval_model(vgg16_classifier, val, criterion, device)
