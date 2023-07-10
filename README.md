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

## Download model parameter
### 1. protbert
Download protbert parameterfile [protbert.tar.gz] on (https://zenodo.org/record/8129221)
Then
```
tar -xzvf protbert.tar.gz
```
Place the address of the fold`protbert` in these two places in file `src.predict_model.py`:
```
self.tokenizer = BertTokenizer.from_pretrained('****/protbert', do_lower_case=False )
self.model = BertModel.from_pretrained('****/protbert')
```
### 2. model trained
If you want to train a model yourself from scratch, please skip this step.<br>
Choose the model you want to use.(Transformer or LSTM)
```
cd policy_transformer
```
or
```
cd policy_LSTM
```
Download the model file you need on (https://zenodo.org/record/8129221).<br>
We have provided model parameter files for TopoProGenerator[model_transformer.pth] and LSTM[model_LSTM.pt] as a reference model.<br>
Modify `"generator_model"` in `config/generate_transformer.json` and locate your model parameter file.

## Train your own model
If you want to use a trained model, skip this step.<br>
Edit `/config/generate_transformer.json` 
name|content
---- | -----
xxx|222





