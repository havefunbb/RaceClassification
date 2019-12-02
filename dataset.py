# by seymayucer
# December 2
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
__all__ = ['CustomDataset']

class CustomDataset(Dataset):
    """
    # Description:
        Basic class for retrieving images and labels
    # Member Functions:
        __init__(self, phase, shape):   initializes a dataset
            meta_data:                  stores image paths and labels
        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset
        __len__(self):                  returns the length of dataset
    """

    def __init__(self, data_root,csv_file,transform):
        self.meta_data = pd.read_csv(csv_file)
        self.race_labels = self.meta_data.race.values
        self.transform =  transform
  

    def __getitem__(self, item):
        path = self.data_path[item]
        image = Image.open(path).convert('RGB')  # (C, H, W)
        image = self.transform(image)
       
        return image, self.race_labels[item]

    def __len__(self):
        return len(self.data_path)