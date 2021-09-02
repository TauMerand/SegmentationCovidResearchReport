from .utils import save_checkpoint

from typing import Optional, OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import initialization as init


import time
from tqdm.notebook import tqdm

class PretrainModel(nn.Module):
  def __init__( self, 
                backbone: str, 
                name: str,
                weights: Optional[str] = None,
                ckpt_state: Optional[str] = None,
              ):
    super().__init__()
    self.name=name
    if ckpt_state is None:
      self.backbone=get_encoder(backbone, 
                                in_channels=3, 
                                depth=5, 
                                weights=weights)
    else:
      print('Loading pretrained model')
      model = PretrainModel(backbone=backbone, name=name)
      model.load_state_dict(ckpt_state, strict=False)
      self.backbone = model.backbone
      del model
        
  @autocast()
  def forward(self, x):
    x = self.backbone(x)[-1] 
    return x


class PretrainClassifier(nn.Module):
  def __init__( self, 
                backbone: str, 
                num_classes: int,
                linear_in_features: int,
                name: str,
                weights: Optional[str] = None,
                ckpt_state: Optional[OrderedDict] = None,
                backbone_state: Optional[OrderedDict] = None,
                
              ):
    super().__init__()
    self.name=name
    if ckpt_state is None:
      model=PretrainModel(backbone=backbone, 
                      weights=weights, 
                      name=name,
                      ckpt_state=backbone_state
                    )
      self.backbone=model.backbone
      del model
      self.fc = nn.Linear(linear_in_features, 2048, bias=True)
      self.classify=nn.Linear(2048, num_classes)
    else:
      model=PretrainClassifier(backbone=backbone, 
                              num_classes=num_classes,
                              linear_in_features=linear_in_features,
                              name=name)
      model.load_state_dict(ckpt_state)
      self.backbone=model.backbone
      self.fc=model.fc
      self.classify=model.classify
      del model
    
      # init.initialize_head(self.fc)
      # init.initialize_head(self.classify)

        
  @autocast()
  def forward(self, x):
    x = self.backbone(x)[-1] 
    # print("Backbone out shape: {}".format(x.shape))
    x=torch.flatten(x, 1)
    # print("Flatten out shape: {}".format(x.shape))
    x = self.fc(x)
    x = F.dropout(x, p=0.5, training=self.training)
    x = F.relu(x)
    x = self.classify(x)
    return x


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
          save_checkpoint(model, optimizer, sub_dir='train', **save_cfg)
          
      loop.set_postfix(loss=np.mean(batch_losses))
  else: #TODO TPU XLA code
    print("TPU")

  return batch_losses

def validate_model(model, 
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

