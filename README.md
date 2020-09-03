# Train CIFAR10,CIFAR100 with Pytorch-lightning
Measure CNN,VGG,Resnet,WideResnet models' accuracy on dataset CIFAR10,CIFAR100 using [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning).

## Requirements
- setup/requirements.txt
```bash
torch 1.5.1
torchvision 0.6.1
pytorch-lightning 0.9.0rc5
tqdm
argparse
pytablewriter
seaborn
enum34
scipy
cffi
sklearn
```

- install requirements using pip
```bash
pip3 install -r setup/requirements.txt
```

## How to run
After you have cloned the repository, you can train each dataset of either cifar10, cifar100 by running the script below.

```bash
python train.py
```

## Implementation Details
- CIFAR10

|   epoch   | learning rate |  weight decay | Optimizer | Momentum |  Nesterov |
|:---------:|:-------------:|:-------------:|:---------:|:--------:|:---------:|
|   0 ~ 20  |      0.1      |     0.0005    |    SGD    |    0.9   |   False   |
|  21 ~ 40  |      0.01     |     0.0005    |    SGD    |    0.9   |   False   |
|  41 ~ 60  |      0.001    |     0.0005    |    SGD    |    0.9   |   False   |

- CIFAR100

|   epoch   | learning rate |  weight decay | Optimizer | Momentum |  Nesterov |
|:---------:|:-------------:|:-------------:|:---------:|:--------:|:---------:|
|   0 ~ 60  |      0.1      |     0.0005    |    SGD    |    0.9   |   False   |
|  61 ~ 120 |      0.01     |     0.0005    |    SGD    |    0.9   |   False   |
| 121 ~ 180 |      0.001    |     0.0005    |    SGD    |    0.9   |   False   |

## Accuracy
Below is the result of the test set accuracy for CIFAR-10, CIFAR-100 dataset training

**Accuracy of models trained on CIFAR10**
| network           | dropout | preprocess | parameters | accuracy(%) |
|:-----------------:|:-------:|:----------:|:----------:|:-----------:|
|       VGG16       |    0    |   meanstd  |     14M    |    91.09    |
|      Resnet-50    |    0    |   meanstd  |     23M    |    92.11    |
| WideResnet 28x10  |   0.3   |   meanstd  |     36M    |    93.61    |
|     Densenet-BC   |    0    |   meanstd  |    769K    |    92.85    |
|      Densenet     |    0    |   meanstd  |    769K    |    93.06    |


**Accuracy of models trained on CIFAR100**
| network           | dropout | preprocess | parameters | accuracy(%) |
|:-----------------:|:-------:|:----------:|:----------:|:-----------:|
|       VGG16       |    0    |   meanstd  |     14M    |    72.79    |
|      Resnet-50    |    0    |   meanstd  |     23M    |    75.80    |
| WideResnet 28x20  |   0.3   |   meanstd  |    145M    |    75.46    |
|     Densenet-BC   |    0    |   meanstd  |    800K    |    72.23    |
|      Densenet     |    0    |   meanstd  |    800K    |    75.58    |
