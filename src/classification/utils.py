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

def save_checkpoint(model,
                    out_name=None,
                    epoch=None,
                    ckpt_dir=None,
                    train_loss=None, 
                    val_loss=None,
                    **extra_args
                    ):
  if ckpt_dir is None:
    ckpt_dir='./{}_ckpt'.format(model.name)
  
  os.makedirs(ckpt_dir, exist_ok=True)

  if out_name is not None:
    ckpt_path=ckpt_dir+'/{}'.format(out_name)
  ckpt_path=ckpt_dir+'/{}'.format(model.name)
  
  if epoch is not None:
    ckpt_path+='_epoch:{}'.format(epoch)
  if train_loss is not None:
    ckpt_path+="_t:{:.3f}".format(train_loss)
  if val_loss is not None:
    ckpt_path+="_v:{:.3f}".format(val_loss)
  
  torch.save(model.state_dict(), ckpt_path+".pt")