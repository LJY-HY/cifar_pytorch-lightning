import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.functional as F

import torchvision.transforms as transforms
import torch.utils.data
import torchvision.models.vgg as VGG
import torchvision.models.resnet as Resnet
from models.WideResnet import Wide_ResNet
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

class CIFAR10_VGG(pl.LightningModule):
    # This Module is based on VGG-16 for dataset CIFAR10
    def __init__(self):
        super(CIFAR10_VGG, self).__init__()
        self.model = VGG.vgg16_bn()
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model.classifier = nn.Sequential(
            nn.Linear(512,10)
        )

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self,batch,batch_idx):
        data, target = batch
        loss = F.cross_entropy(self.forward(data), target)
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1), 'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

    def validation_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'val_loss':loss,'correct':correct}

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'val_loss':avg_loss}
        print('Validation accuracy : ',sum_correct/10000,'\n\n') # self.arg.validation_size
        return {'avg_val_loss':avg_loss, 'log':tensorboard_logs}    

    def test_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'test_loss':F.cross_entropy(output,target), 'correct':correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'test_loss': avg_loss}
        print('Test accuracy :',sum_correct/10000,'\n')
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

class CIFAR100_VGG(pl.LightningModule):
    # This Module is based on VGG-16 for dataset CIFAR100
    def __init__(self):
        super(CIFAR100_VGG, self).__init__()
        self.model = VGG.vgg16_bn()
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model.classifier = nn.Sequential(
            nn.Linear(512,100)
        )

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output,target)
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1), 'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

    def validation_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output,target)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'val_loss':loss,'correct':correct}

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'val_loss':avg_loss}
        print('Validation accuracy : ',sum_correct/10000,'\n\n')
        return {'avg_val_loss':avg_loss, 'log':tensorboard_logs}    

    def test_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'test_loss':F.cross_entropy(output,target), 'correct':correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'test_loss': avg_loss}
        print('Test accuracy :',sum_correct/10000,'\n')
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

class CIFAR10_Resnet(pl.LightningModule):
    # This Module is based on Resnet-50 for dataset CIFAR10
    def __init__(self):
        super(CIFAR10_Resnet, self).__init__()
        self.model = Resnet.ResNet(Resnet.Bottleneck,[3,4,6,3],num_classes=10)
        self.model.inplanes=64
        self.model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)
        self.model.linear = nn.Linear(512*Resnet.Bottleneck.expansion, 10)
        del self.model.maxpool
        self.model.maxpool = lambda x : x

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output,target)
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1), 'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

    def validation_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output,target)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'val_loss':loss,'correct':correct}

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'val_loss':avg_loss}
        print('Validation accuracy : ',sum_correct/10000,'\n\n')
        return {'avg_val_loss':avg_loss, 'log':tensorboard_logs}    

    def test_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'test_loss':F.cross_entropy(output,target), 'correct':correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'test_loss': avg_loss}
        print('Test accuracy :',sum_correct/10000,'\n')
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

class CIFAR100_Resnet(pl.LightningModule):
    # This Module is based on Resnet-50 for dataset CIFAR100
    def __init__(self):
        super(CIFAR100_Resnet, self).__init__()
        self.model = Resnet.ResNet(Resnet.Bottleneck,[3,4,6,3],num_classes=100)
        self.model.inplanes=64
        self.model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)
        self.model.linear = nn.Linear(512*Resnet.Bottleneck.expansion, 100)
        del self.model.maxpool
        self.model.maxpool = lambda x : x
    
    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output,target)
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1), 'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

    def validation_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output,target)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'val_loss':loss,'correct':correct}

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'val_loss':avg_loss}
        print('Validation accuracy : ',sum_correct/10000,'\n\n')
        return {'avg_val_loss':avg_loss, 'log':tensorboard_logs}                    

    def test_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'test_loss':F.cross_entropy(output,target), 'correct':correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'test_loss': avg_loss}
        print('Test accuracy :',sum_correct/10000,'\n')
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

class CIFAR10_WideResnet(pl.LightningModule):
    # This Module is based on WideResNet28-10 for dataset CIFAR10
    def __init__(self):
        super(CIFAR10_WideResnet, self).__init__()
        self.model = Wide_ResNet(28,10,0.3,10)

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output,target)
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}

    def configure_optimizers(self):
        optimizer=optim.SGD(self.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
        lr_scheduler = {'scheduler':optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,40],gamma=0.1),'interval':'epoch'}
        return [optimizer], [lr_scheduler]

    def validation_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output,target)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'val_loss':loss,'correct':correct}

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'val_loss':avg_loss}
        print('Validation accuracy : ',sum_correct/10000,'\n\n') 
        return {'avg_val_loss':avg_loss, 'log':tensorboard_logs}    

    def test_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'test_loss':F.cross_entropy(output,target), 'correct':correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'test_loss': avg_loss}
        print('Test accuracy :',sum_correct/10000,'\n')
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}


class CIFAR100_WideResnet(pl.LightningModule):
    # This Module is based on WideResNet 28-20 for dataset CIFAR-100
    def __init__(self):
        super(CIFAR100_WideResnet, self).__init__()
        self.model = Wide_ResNet(28,20,0.3,100)

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output,target)
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}

    def configure_optimizers(self):
        optimizer=optim.SGD(self.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
        lr_scheduler = {'scheduler':optim.lr_scheduler.MultiStepLR(optimizer,milestones=[60,120],gamma=0.1),'interval':'epoch'}
        return [optimizer], [lr_scheduler]

    def validation_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output,target)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'val_loss':loss,'correct':correct}

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'val_loss':avg_loss}
        print('Validation accuracy : ',sum_correct/10000,'\n\n')
        return {'avg_val_loss':avg_loss, 'log':tensorboard_logs}    

    def test_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'test_loss':F.cross_entropy(output,target), 'correct':correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'test_loss': avg_loss}
        print('Test accuracy :',sum_correct/10000,'\n')
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}