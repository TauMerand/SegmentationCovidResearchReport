import numpy as np 
import cv2 as cv
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class ChexpertDataset(Dataset):
    def __init__(self, dataframe, image_dir, image_shape , mode, classes, df_frac=1):
        super().__init__()
        assert mode in ('train', "test")
        self.mode=mode
        self.df=dataframe.sample(frac=df_frac).reset_index(drop=True) 
        self.labels=classes
        self.img_dir=image_dir
        self.img_size=image_shape
        self.transform = albu.Compose([
                albu.Resize(self.img_size[0], self.img_size[1]),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path='{}/{}'.format(self.img_dir, self.df.loc[0, 'Path'])
        img = cv.imread(img_path) # Not reading as cv2.IMREAD_GRAYSCALE so don't need to stack
        img = self.transform(image=img)['image']
        img_label = torch.Tensor(self.df.loc[idx, self.labels])
        img_label[img_label.isnan()]=0
        img_label[img_label==-1]=0
        return img, img_label