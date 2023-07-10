# TopoProGenerator
Generating protein sequences with specifying topological structures 
FIG

## Requirements
### 1. Install Pytorch(If the GPU is already usable, skip this step)
####1.1 Obtain CUDA version<br>
```
nvidia-smi
```
According to the CUDA version, install a compatible version of Pytorch on the [Pytorch website]（https://pytorch.org/）
#### 2. Check if pytorch installation was successful
```
python
import torch
torch.cuda.is_available()
```
if return `True`, pytorch is already installed.


