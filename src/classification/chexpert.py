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
                **save_args):

  save_freq=save_args.pop("save_freq", len(train_loader)//2)
  start_time=save_args.pop("start_time", time.time())
  time_out=save_args.pop("time_out", 9*60*60)
  threshold=save_args.pop("time_threshold", 0.8)

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
        if time.time() - start_time > threshold*time_out:
          save_checkpoint(model, train_loss=np.mean(batch_losses), **save_args)      
      loop.set_postfix(loss=np.mean(batch_losses))
  else: #TODO TPU XLA code
    print("TPU")

  return np.mean(batch_losses)


def eval_model(model, val_loader, criterion, device):
  model.eval()
  val_loss = 0.0
  if device.type == 'cuda' or device.type == 'cpu':
    model.to(device)
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


def vgg16_classifier(loader_cfg: Optional[Dict[str, str]] = None,
                      saving_cfg: Optional[Dict[str, str]] = None,
                      weights: Optional[str] = "imagenet",
                      epochs: Optional[int] = 1,
                      device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    ):

  train_loader, val_loader = ChexpertLoader(**loader_cfg)

  vgg16 = PretrainClassifier(backbone="vgg16", 
                              weights=weights,
                              num_classes=14,
                              linear_in_features=512*12*10,
                            )
  vgg16.to(device)

  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(vgg16.parameters(),
                                lr=0.001
                              )
  
  min_val_loss=np.inf
  for i in trange(epochs, desc='Epochs Outer'):
    curr_loss=train_model(vgg16,
                          train_loader, 
                          criterion, 
                          optimizer, 
                          device,
                          **saving_cfg)

    # if time.time() - start_time > 0.9*time_out:
      

  #   curr_loss=eval_model(vgg16_classifier, val, criterion, device)

  #   if curr_loss<min_val_loss:
  #     filename=ckpt_dir+"/vgg_epoch_{}_train_{:.3f}_min_val_{:.3f}.pt".format(i,
  #                                                                 t_loss,
  #                                                                 curr_loss)
  #     # print("Saving Model with validation loss: {:.3f}".format(curr_loss))
  #     torch.save(vgg16_classifier.state_dict, filename)
  #     min_val_loss=curr_loss
  # filename=ckpt_dir+'/completed_vgg_train_{:.3f}_val_{.3f}.pt'
  # torch.save(vgg16_classifier, filename)
