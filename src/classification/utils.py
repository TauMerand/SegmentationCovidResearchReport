import os
import numpy as np
import torch

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