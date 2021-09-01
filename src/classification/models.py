from typing import Optional

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
                ckpt_path: Optional[str] = None,
              ):
    super().__init__()
    self.name=name
    if ckpt_path is None:
      self.backbone=get_encoder(backbone, 
                                in_channels=3, 
                                depth=5, 
                                weights=weights)
      model=None
    else:
      print('Loading pretrained model from: {}'.format(ckpt_path))
      model = PretrainModel(backbone=backbone, 
                            weights=None,
                            ckpt_path=None, 
                            name = name
                            )
      model.load_state_dict(torch.load(ckpt_path), strict=False)
      self.backbone = model.backbone
      del model.backbone
    return model
        
  @autocast()
  def forward(self, x):
    x = self.backbone(x)[-1] 
    return x


class PretrainClassifier(PretrainModel):
  def __init__( self, 
                backbone: str, 
                num_classes: int,
                linear_in_features: int,
                name: str,
                weights: Optional[str] = None,
                ckpt_path: Optional[str] = None,
                pretrained_backbone: Optional[str] = None,
                
              ):
    model=super().__init__(backbone=backbone, 
                            weights=weights, 
                            name=name,
                            ckpt_path=pretrained_backbone
                          )
    if ckpt_path is None and model is None:
      self.fc = nn.Linear(linear_in_features, 2048, bias=True)
      self.classify=nn.Linear(2048, num_classes)
      # init.initialize_head(self.fc)
      # init.initialize_head(self.classify)
    else:
      self.fc=model.fc
      self.classify=model.classify
      del model

        
  @autocast()
  def forward(self, x):
    x = self.backbone(x)[-1] 
    x=torch.flatten(x, 1)
    x = self.fc(x)
    x = F.dropout(x, p=0.5, training=self.training)
    x = F.relu(x)
    x = self.classify(x)
    return x


