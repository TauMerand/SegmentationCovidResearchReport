from typing import Optional, Dict

from .utils import save_checkpoint
from .models import PretrainClassifier
from .data import ChexpertLoader

import numpy as np
import time
import os

# from tqdm import tqdm, trange
from tqdm.notebook import trange, tqdm


import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torch.cuda.amp import autocast

# from warmup_scheduler import GradualWarmupScheduler



def train_model(model, 
                train_loader, 
                criterion, 
                optimizer, 
                device, 
                **save_cfg):

  save_freq=save_cfg.pop("save_freq", len(train_loader)//2)
  time_sensitive=save_cfg.pop("time_sensitive", False)
  start_time=save_cfg.pop("start_time", time.time())
  time_out=save_cfg.pop("time_out", 9*60*60)
  threshold=save_cfg.pop("time_threshold", 0.8)

  model.train()
  batch_losses = []
  if device.type == 'cuda' or device.type == 'cpu':
    loop = tqdm(train_loader, desc='Train Inner')
    for i, (images, labels) in enumerate(loop):
      images = images.to(device)
      labels = labels.to(device)
      with torch.cuda.amp.autocast(): 
        outputs = model(images)
        loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      batch_losses.append(loss.item())
      if i % save_freq == 0:
        if not time_sensitive or time.time() - start_time > threshold*time_out:
          print("Saving Checkpoint i = {}".format(i))
          save_checkpoint(model, optimizer, sub_dir='train', **save_cfg):
          
      loop.set_postfix(loss=np.mean(batch_losses))
  else: #TODO TPU XLA code
    print("TPU")

  return batch_losses


def eval_model(model, 
                val_loader,
                criterion, 
                device):
  model.eval()
  val_loss = 0.0
  if device.type == 'cuda' or device.type == 'cpu':
    for images, labels in tqdm(val_loader, desc="Val Inner"):
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(), torch.no_grad():
          outputs = model(images)
          loss = criterion(outputs.float(), labels)
                
        val_loss += loss.item() * images.size(0)
  else: #TODO TPU Code
    print('TPU')  
  return val_loss / len(val_loader.dataset)


def vgg16_classifier(loader_cfg: Optional[Dict[str, str]] = {},
                      save_cfg: Optional[Dict[str, str]] = {},
                      weights: Optional[str] = "imagenet",
                      epochs: Optional[int] = 1,
                      device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                      ckpt: Optional[str] = None
                    ):

  train_loader, val_loader = ChexpertLoader(**loader_cfg)

  checkpoint = torch.load(ckpt)
  
  vgg16 = PretrainClassifier(backbone="vgg16", 
                              weights=weights,
                              num_classes=14,
                              linear_in_features=512*12*10,
                              name="vgg16_{}".format(weights),
                              ckpt_state=checkpoint['state_dict']
                            )
  vgg16.to(device)

  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(vgg16.parameters(),
                                lr=0.001
                              )
  optimizer.load_state_dict(checkpoint['optimizer'])
        
  epochs+=checkpoint['epoch']
  min_val_loss=np.inf
  for i in trange(checkpoint['epoch'], epochs, desc='Epochs Outer'):
    train_losses=train_model(vgg16,
                          train_loader, 
                          criterion, 
                          optimizer, 
                          device,
                          **save_cfg)      
    val_loss=eval_model(vgg16, val_loader, criterion, device)

    if val_loss<min_val_loss:
      save_checkpoint(vgg16, epoch=i, train_loss=np.mean(train_losses), val_loss=val_loss, **save_cfg) 
      min_val_loss=val_loss
  save_checkpoint(vgg16, out_name="completed", epoch=epochs, train_loss=np.mean(train_losses), val_loss=val_loss, **save_cfg) 
  return vgg16, train_loader, val_loader
