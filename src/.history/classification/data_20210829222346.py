import numpy as np 
import cv2 as cv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class ExternalImageDataset(Dataset):
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

def ChexpertLoader(train_path = "../input/chexpert-082021/CheXpert-v1.0-small/train.csv", 
                  val_path = "../input/chexpert-082021/CheXpert-v1.0-small/valid.csv",
                  image_dir='../input/chexpert-082021',
                  image_size=(320, 390),
                  train_frac=1,
                  val_frac=1,
                  batch_size=64,
                  num_workers=4,
                  loader=DataLoader):

  train_df = pd.read_csv(train_path)
  valid_df = pd.read_csv(val_path)
  dst_classes = train_df.loc[:, "No Finding":].columns.to_list()
  
  train_dataset = ExternalImageDataset( dataframe=train_df, 
                                        image_dir=image_dir, 
                                        image_shape=image_size, 
                                        mode='train', 
                                        classes=dst_classes, 
                                        df_frac=train_frac)

  val_dataset = ExternalImageDataset( dataframe=valid_df, 
                                      image_dir=image_dir, 
                                      image_shape=image_size, 
                                      mode='test',
                                      classes=dst_classes,
                                      df_frac=val_frac)

  train_loader = loader(train_dataset, 
                        batch_size=batch_size, 
                        num_workers=num_workers)

  val_loader = loader(val_dataset, 
                      batch_size=batch_size, 
                      num_workers=num_workers)

  print('TRAIN: {} | VALID: {}'.format(len(train_loader.dataset), 
                                      len(val_loader.dataset)))
  return train_loader, val_loader
