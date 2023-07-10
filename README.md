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
If return `True`, pytorch is already installed.

### Install other requirements
```
pip install -r requirements
```

## model parameter
### 1. protbert
Down protbert parameterfile [protbert.tar.gz] on (https://zenodo.org/record/8129221)
Then
```
tar -xzvf protbert.tar.gz
```
Place the address of the fold`protbert` in these two places in file `src.predict_model.py`:
```
self.tokenizer = BertTokenizer.from_pretrained(__'../transformer/protbert'__, do_lower_case=False )
self.model = BertModel.from_pretrained(__'../transformer/protbert'__)
```

