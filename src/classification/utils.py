import os
import numpy as np
import torch
import time

def all_the_seeds(seed=42):
#     random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

cfg={
    'encoder_name': 'vgg16',
    'encoder_weights': 'imagenet',
    'in_features': 2048,
    'workers': 4,

    'chexpert_image_size': (320, 390),
    'chexpert_batch_size': 96,
    'chexpert_init_lr': 0.0001,
    'chexpert_epochs': 2,
}    

def save_checkpoint(model,
                    epoch=None,
                    ckpt_dir=None,
                    train_loss=None, 
                    val_loss=None,
                    **extra_args
                    ):
  if ckpt_dir is None:
    ckpt_dir='./{}_ckpt'.format(model.__class__.__name__)

  ckpt_path=ckpt_dir+'/{}'.format(model)
  if epoch is not None:
    ckpt_path+='_epoch:{}'.format(epoch)
  if train_loss is not None:
    ckpt_path+="_t:{:.3f}".format(train_loss)
  if val_loss is not None:
    ckpt_path+="_v:{:.3f}".format(val_loss)
  
  torch.save(model.state_dict, ckpt_path+".pt")