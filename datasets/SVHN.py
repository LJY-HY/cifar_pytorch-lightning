import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

import pytorch_lightning as pl

class SVHNDataModule(pl.LightningDataModule):
    def __init__(self,batch_size=64):
        super().__init__()
        self.mean = [129.3/255, 124.1/255, 112.4/255]
        self.std = [68.2/255, 65.4/255.0, 70.4/255.0]
        
        self.transform = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        self.batch_size=batch_size

    def prepare_data(self):
        datasets.SVHN(root='./workspace/datasets/SVHN', split='train', transform=self.transform, download=True)
        datasets.SVHN(root='./workspace/datasets/SVHN', split ='extra',transform=self.transform, download=True)
        datasets.SVHN(root='./workspace/datasets/SVHN', split ='test',transform=self.transform, download=True)

    def setup(self, stage=None):
        self.SVHN_test = datasets.SVHN(root='./workspace/datasets/SVHN', split='test',transform=self.transform, download=True)
        self.SVHN_train = datasets.SVHN(root='./workspace/datasets/SVHN', split='train',transform=self.transform, download=True)
        self.SVHN_val = datasets.SVHN(root='./workspace/datasets/SVHN', split='extra',transform=self.transform, download=True)

    def train_dataloader(self):
        SVHN_train = DataLoader(self.SVHN_train, batch_size=self.batch_size, shuffle=True, num_workers=8)
        return SVHN_train

    def val_dataloader(self):
        SVHN_val = DataLoader(self.SVHN_val, batch_size=self.batch_size, shuffle=False, num_workers=8)
        return SVHN_val

    def test_dataloader(self):
        return DataLoader(self.SVHN_test, batch_size=self.batch_size, shuffle=False, num_workers=8)
