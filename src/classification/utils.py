import os
import numpy as np
import torch
import time
import glob

def all_the_seeds(seed=42):
#     random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_checkpoint(model,
                    optimizer,
                    ckpt_num=None,
                    sub_dir=None,
                    epoch=None,
                    ckpt_dir=None,
                    train_loss=None, 
                    val_loss=None,
                    **extra_args
                    ):
  if ckpt_dir is None:
    ckpt_dir='./checkpoints'.format(model.name)
  if sub_dir is not None:
    ckpt_dir+="/{}".format(sub_dir)

  os.makedirs(ckpt_dir, exist_ok=True)
  ckpt_path=ckpt_dir+'/{}'.format(model.name)

  if epoch is not None:
    ckpt_path+='_epoch:{}'.format(epoch)
  if train_loss is not None:
    ckpt_path+="_t:{:.3f}".format(train_loss)
  if val_loss is not None:
    ckpt_path+="_v:{:.3f}".format(val_loss)
  if ckpt_num is not None:
    ckpt_path+='.pt{}'.format(ckpt_num)

  state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
  
  torch.save(state, ckpt_path)

# def save_train_checkpoint(model, 
#                           optimizer,
#                           ckpt_num,
#                           ckpt_dir=None,
#                           **extra_args):
#   if ckpt_dir is None:
#     ckpt_dir='./{}_train_ckpt'.format(model.name)
#     if ckpt_num == None:
#       ckpt_num=0
#   # else:
#   os.makedirs(ckpt_dir, exist_ok=True)
#   ckpt_path=ckpt_dir+'/{}.pt{}'.format(model.name, ckpt_num)
  
#   torch.save(state, ckpt_path)

  
#   os.makedirs(ckpt_dir, exist_ok=True)