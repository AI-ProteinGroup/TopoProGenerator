# TopoProGenerator
Generating protein sequences with specified topological structures.
![frame](https://github.com/AI-ProteinGroup/TopoProGenerator/blob/main/frame1.png)


## Download project
```
git clone https://github.com/AI-ProteinGroup/TopoProGenerator.git
cd TopoProGenerator
```

## Install Requirements
### 1. Install Pytorch(if GPU is usable, skip this step)
#### 1.1 know your CUDA version<br>
```
nvidia-smi
```
According to the CUDA version, install a compatible version of Pytorch on the [Pytorch website]（https://pytorch.org/）
#### 1.2 Check if pytorch installation was successful
```
python
import torch
torch.cuda.is_available()
```
If return `True`, pytorch is already installed.

### 2. Install other requirements
```
pip install -r requirements.txt
```

## Download model parameter
### 1. Protbert
Download protbert parameterfile [protbert.tar.gz] on (https://zenodo.org/record/8129221)<br>
Then
```
tar -xzvf protbert.tar.gz
```
Place the address of the fold`protbert` in the below two places in file `policy_transformer/src/predict_model.py` or `policy_LSTM/src/predict_model.py`:
```
self.tokenizer = BertTokenizer.from_pretrained('****/protbert', do_lower_case=False)
self.model = BertModel.from_pretrained('****/protbert')
```
### 2. Model trained
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

## Train your own model(Taking Transformer as an example)
If you want to use a trained model, skip this step.<br>
### 1. Prepare dataset
You need to process the sequence data set into the form like (i means 'HHH')
```
iDEEERRVEELIEEARELEKRNPEEARKVLEEAYELAKRINDPLLEEVEKLLRRLR
iSEHEERIRELLERARRIPDKEEARRLVEEAIRIAEENNDEELLKKAREILEEIKR
```
save it as `*.csv`, which is the dataset for pretraining.<br>
And process the ori sequence sets like
```
DEEERRVEELIEEARELEKRNPEEARKVLEEAYELAKRINDPLLEEVEKLLRRLR
SEHEERIRELLERARRIPDKEEARRLVEEAIRIAEENNDEELLKKAREILEEIKR
```
save it as `*.txt` and `*.csv`, which are the dataset for finetuning.<br>
For TPG, both the pretraining dataset and fine-tuning dataset are in `./data`.
### 2. Pretrain
!!!model parameters need to be consistent during pretraining, fine-tuning and generate.(such as `tgt_len`, `d_embed`, `n_layers` and so on).<br>
Edit `/config/pretrain_transformer.json` 
name|content
---- | -----
datasets|Address of the dataset used for pretraining
datasets_col|The column where the protein sequence is located (starting from 0)
save_addr|Address of output model file

Start pretraining
```
python pretrain_transformer.py --config ./config/pretrain_transformer.json
```

### 3. Fine-tune
Edit `/config/fine-tuning_transformer.json` (model parameters need to be consistent with the pretraining)
name|content
---- | -----
fine_tuning_datasets|Address of the dataset used for pretraining`.csv`
datasets_col|The column where the protein sequence is located (starting from 0)
truth_seq_datasets|Address of the dataset used for pretraining`.txt`
prime_str|Topology labels specified for generated sequnece
generator_model|Address of pretrained model
num_epochs|Total epoch of fine-tuning
g_epoch|Epoch of Generative model training in each round of fine-tuning
d_epoch|Epoch of Discriminative model training in each round of fine-tuning
fake_data_num|Number of generated sequences for Discriminative model training
predictor_score_up|Weights of stable sequences
predictor_score_up|Weights of unstable sequences
save_addr|Address of output model file

Start fine-tuning
```
python fine-tuning_transformer.py --config ./config/fine-tuning_transformer.json
```
After each epoch of fine-tuning, the model will generate 20000 sequences simultaneously.

## Generate sequences
Edit `/config/generate_transformer.json` (model parameters need to be consistent with the pretraining or fine-tuning)
name|content
---- | -----
prime_str|topology labels specified for generated sequnece
generator_model|Address of model
num_seq|Number of generated sequences
min_length|Minimum length of generated sequence
max_length|Maximum length of generated sequence which should be smaller than `tge_len`
seq_save|Address for generating sequence file

Generate sequences
```
python generate_transformer.py --config ./config/generate_transformer.json
```












