from typing import Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import initialization as init


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
    if ckpt_state is None:
      self.backbone=PretrainModel(backbone=backbone, 
                      weights=weights, 
                      name=name,
                      ckpt_state=backbone_state
                    )
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
    x=torch.flatten(x, 1)
    print(x.shape)
    x = self.fc(x)
    x = F.dropout(x, p=0.5, training=self.training)
    x = F.relu(x)
    x = self.classify(x)
    return x


