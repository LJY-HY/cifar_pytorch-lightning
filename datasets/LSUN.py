import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

import pytorch_lightning as pl

class LSUNDataModule(pl.LightningDataModule):
    # This DataModule is usded only for out-of-distribution dataset.
    # Therefore there are no seperated train/val/test dataset/
    def __init__(self,batch_size=64):
        super().__init__()
        # these mean and std are not LSUN mean/std
        self.mean = [125.3/255, 123.0/255, 113.9/255]
        self.std = [63.0/255, 62.1/255.0, 66.7/255.0]
        
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        self.batch_size=batch_size

    def prepare_data(self):
        pass
        
    def setup(self, stage=None):
        self.lsun_dataset = datasets.ImageFolder(root='./workspace/datasets/LSUN', transform=self.transform)
        
    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return DataLoader(self.lsun_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)


class LSUN_resizeDataModule(pl.LightningDataModule):
    # This DataModule is usded only for out-of-distribution dataset.
    # Therefore there are no seperated train/val/test dataset/
    def __init__(self,batch_size=64):
        super().__init__()
        # these mean and std are not LSUN mean/std
        self.mean = [125.3/255, 123.0/255, 113.9/255]
        self.std = [63.0/255, 62.1/255.0, 66.7/255.0]
        
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        self.batch_size=batch_size

    def prepare_data(self):
        pass
        
    def setup(self, stage=None):
        self.lsun_resize_dataset = datasets.ImageFolder(root='./workspace/datasets/LSUN_resize', transform=self.transform)
        
    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return DataLoader(self.lsun_resize_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)