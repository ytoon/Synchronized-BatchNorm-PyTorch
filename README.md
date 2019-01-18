# Synchronized BatchNorm in PyTorch 1.0
This **GPU** version is compatible with PyTorch 1.0. And, it is inspired by [**PyTorch-Encoding**](https://github.com/zhanghang1989/PyTorch-Encoding). 
## Installation
1. [PyTorch](https://pytorch.org/)
2. Synchronized BatchNorm PyTorch
```
bash compile.sh
```
Then, import synchronized batchnorm in your code as follows:
```
from syncbn import SyncBN
```
## Demo
1. CIFAR-10
```
python3 cifar10.py
```