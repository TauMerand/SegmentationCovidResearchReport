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



def train_model(model, train_loader, criterion, optimizer, device, save_freq=None, start_time=None, time_out=None, ckpt_dir=None):
  model.train()
  train_loss = []
  if device.type == 'cuda' or device.type == 'cpu':
    loop = tqdm(train_loader)
    for i, (images, labels) in enumerate(loop):
      images = images.to(device)
      labels = labels.to(device)
      with torch.cuda.amp.autocast(): 
        outputs = model(images)
        loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())
      if save_freq and i % save_freq == 0:
        if time.time() - start_time > 0.8*time_out:
          ave_loss=np.mean(train_loss)
          print("Saving Model with loss: {:02d}".format(ave_loss))
          filename=ckpt_dir+'/vgg16_train_{}_{:02d}.pt'.format(i, ave_loss)
          torch.save(model.state_dict(), filename)
      # loop.set_postfix(loss=np.mean(train_loss))
  else: #TODO TPU XLA code
    print("TPU")

  return np.mean(train_loss) # = np.mean(train_loss)


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


def vgg16_classifier(epochs, device, ckpt_dir, start_time, save_freq, time_out):
  train, val = ChexpertLoader()
  vgg16_classifier = PretrainClassifier(backbone="vgg16", 
                                        weights="imagenet",
                                        num_classes=14,
                                        linear_in_features=512*12*10,
                                        )
  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam( vgg16_classifier.parameters(),
                                lr=0.001)
  min_val_loss=np.inf
  loop = tqdm(epochs)
  for i in loop:
    t_loss=train_model(vgg16_classifier, train, criterion, optimizer, device,
                      save_freq, start_time, time_out, ckpt_dir)
    if time.time() - start_time > 0.9*time_out:
      filename=ckpt_dir+"/vgg_epoch_{}_train_{}.pt".format(i, t_loss)
      torch.save(vgg16_classifier.state_dict, filename)
    curr_loss=eval_model(vgg16_classifier, val, criterion, device)
    if curr_loss<min_val_loss:
      filename=ckpt_dir+"/vgg_epoch_{}_train_{}_min_val_{}.pt".format(i, t_loss,
                                                              curr_loss)
      torch.save(vgg16_classifier.state_dict, filename)
  filename=ckpt_dir+'/completed_vgg.pt'
  torch.save(vgg16_classifier, filename)
