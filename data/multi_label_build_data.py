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

# Clean dataset
data = pd.read_csv('sample_labels.csv')
data['lables'] = data['Finding Labels'].str.split('|')
data['Image Index'] = './sample/images/' + data['Image Index']

# chen = []
# for path_img in data['Image Index'].values:
#     img = Image.open(path_img)
#     chen.append(transforms.ToTensor()(img).shape)

# bad_img = data.loc[pd.DataFrame(chen)[0] > 1, 'Image Index'].values

# for path_img in bad_img:
#     img = plt.imread(path_img)
#     img = img[:, :, 0]
#     img = Image.fromarray(np.uint8(img * 255), 'L')
#     img.save(path_img,"PNG")
    

# Dtaset
class MultiDataset(Dataset):
    def __init__(self, data, transform=None, bad_img=[]):
        self.data = data
        self.transform = transform
        self.labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                        'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
                        'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
                        'Pneumothorax']
        
        self.bad_img = bad_img
        
    def __getitem__(self, idx):
        path_img = self.data.loc[idx, 'Image Index']
        
        label = self.data.loc[idx, 'lables']
        label = [self.labels.index(_) for _ in label]
        labels = np.zeros(len(self.labels))
        labels[label] = 1
        labels = torch.tensor(labels)
        
        
        img = Image.open(path_img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, labels.squeeze(0).float()
    
    def __len__(self):
        return self.data.shape[0]

# --- Train/Test Dataset ---

train = data.loc[:int(0.8*data.shape[0]), :]
test = data.loc[int(0.8*data.shape[0])+1:, :].reset_index(drop=True)

def build_dataloader(transfrom, BATCH_SIZE):
    trainset = MultiDataset(train, transfrom)
    train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    testset = MultiDataset(test, transfrom)
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    num_class = len(['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                        'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
                        'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
                        'Pneumothorax'])
    return train_dataloader, test_dataloader, len(trainset), len(testset), num_class

