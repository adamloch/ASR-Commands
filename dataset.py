import torch
import sys
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data.dataset import Dataset
import pandas as pd
import os
import wavio
import random
import numpy as np
import torch
import librosa
from PIL import Image


class ASR_Dataset(Dataset):
    def __init__(self, path_csv, png_path):
        self.dataset = pd.read_csv(png_path + path_csv, skiprows = 0)
        
        print(self.dataset.head())
        self.png_path = png_path
        self.labels = self.dataset['label'].unique().tolist()
        self.dataset = self.dataset.sample(frac=1)

    def __getitem__(self, index):
        sample = self.dataset.iloc[[index]]
        image = Image.open(self.png_path + sample['path'].values[0])
        label = self.labels.index(sample['label'].values[0])
        image = image.convert('RGB')
        image = np.array(image)
        image = torch.from_numpy(image).float().transpose(0,2)
        #print(image)
        #print(label)
        return (image, label)

    def __len__(self):
        return self.dataset.shape[0]
