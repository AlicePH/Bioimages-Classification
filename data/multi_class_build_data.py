import random
import os
import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from PIL import Image


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything()


# --- Train Dataset ---

path_train = './dataset2-master/dataset2-master/images/TRAIN/'
train = pd.DataFrame(columns=['img_id', 'label'])

for label in os.listdir(path_train):
    img_id = os.listdir(path_train + label)

    df = pd.DataFrame(img_id, columns=['img_id'])
    df['img_id'] = os.path.join(path_train, label)+ '/' + df['img_id']
    
    df['label'] = os.listdir(path_train).index(label)
    train = pd.concat([train, df])
        
        
clean_mask = train['img_id'].apply(lambda x: x.split('/')[-1] =='.ipynb_checkpoints')
train = train.loc[~clean_mask, :].reset_index(drop=True)


# --- Test Dataset ---

path_test = './dataset2-master/dataset2-master/images/TEST/'
test = pd.DataFrame(columns=['img_id', 'label'])

for label in os.listdir(path_test):
    img_id = os.listdir(path_test + label)

    df = pd.DataFrame(img_id, columns=['img_id'])
    df['img_id'] = os.path.join(path_test, label)+ '/' + df['img_id']
    
    df['label'] = os.listdir(path_test).index(label) 
    test = pd.concat([test, df])
    
clean_mask = test['img_id'].apply(lambda x: x.split('/')[-1] =='.ipynb_checkpoints')
test = test.loc[~clean_mask, :].reset_index(drop=True)

# --- Dataset ---
class ClassDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, idx):
        path_img = self.data.loc[idx, 'img_id']
        label = self.data.loc[idx, 'label']
        
        img = Image.open(path_img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return self.data.shape[0]



def build_dataloader(transfrom, BATCH_SIZE):
    trainset = ClassDataset(train, transfrom)
    train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    testset = ClassDataset(test, transfrom)
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    num_class = train.label.unique()
    num_class = len(num_class)
    return train_dataloader, test_dataloader, len(trainset), len(testset), num_class

