from typing import Optional, Dict

from .utils import save_checkpoint
from .models import PretrainClassifier, train_model, validate_model
from .data import ChexpertLoader

import numpy as np


from tqdm.notebook import trange


import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torch.cuda.amp import autocast

# from warmup_scheduler import GradualWarmupScheduler



def vgg16_classifier(loader_cfg: Optional[Dict[str, str]] = {},
                      save_cfg: Optional[Dict[str, str]] = {},
                      weights: Optional[str] = "imagenet",
                      epochs: Optional[int] = 1,
                      device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                      model_ckpt: Optional[str] = None,
                      optim_ckpt: Optional[str] = None,
                    ):
  model_state=torch.load(model_ckpt) if model_ckpt is not None else None

  train_loader, val_loader = ChexpertLoader(**loader_cfg)
  
  vgg16 = PretrainClassifier(backbone="vgg16", 
                              weights=weights,
                              num_classes=14,
                              linear_in_features=512*12*10,
                              name="vgg16_{}".format(weights),
                              ckpt_state=model_state
                            )

  vgg16.to(device)

  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(vgg16.parameters(),
                                lr=0.001
                              )
  if optim_ckpt is not None:
    optimizer.load_state_dict(torch.load(optim_ckpt))

  min_val_loss=np.inf
  for i in trange(epochs, desc='Epochs Outer'):
    train_losses=train_model(vgg16,
                          train_loader, 
                          criterion, 
                          optimizer, 
                          device,
                          **save_cfg)      
    val_loss=validate_model(vgg16, val_loader, criterion, device)

    if val_loss<min_val_loss:
      save_checkpoint(vgg16, sub_dir="min_vals", val_loss=val_loss, **save_cfg) 
      min_val_loss=val_loss
  save_checkpoint(vgg16, out_name="completed", epoch=epochs, train_loss=np.mean(train_losses), val_loss=val_loss, **save_cfg) 
  return vgg16, train_loader, val_loader
