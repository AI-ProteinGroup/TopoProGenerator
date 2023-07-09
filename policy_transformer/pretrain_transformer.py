import os
import math
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import argparse
import json
sys.path.append('./src/')
import src.generator_transformer as Generator

use_cuda = torch.cuda.is_available()
tokens = ['/', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S', 'T', 'C', 'P', 'U', 'O']

'''Load config'''
parser = argparse.ArgumentParser(description="you should add those parameter for pretrain")
parser.add_argument("--config", dest="config", help="path to the JSON config file", required=True)
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)

if (not os.path.exists(config["save_addr"])):
    os.mkdir(config["save_addr"])

'''Load the generator config'''
config_Generator = Generator.Config(pro_vocab_size=len(tokens), use_cuda=use_cuda, tgt_len=config["tgt_len"],
                         d_embed=config["d_embed"], d_ff=config["d_ff"], d_k=config["d_k"], d_v=config["d_v"], n_layers=config["n_layers"], n_heads=config["n_heads"])

'''Define Dataset'''
class MyDataSet(Data.Dataset):
    def __init__(self, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()

        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs


    def __len__(self):
        return len(self.dec_inputs)

    def __getitem__(self, idx):
        return self.dec_inputs[idx],self.dec_outputs[idx]


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

'''Define training and evaluating'''
def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    print(loader.dataset)
    for it, (dec_inputs_batch, dec_outputs_batch) in enumerate(loader):
        dec_inputs_batch, dec_outputs_batch = dec_inputs_batch.cuda(), dec_outputs_batch.cuda()
        optimizer.zero_grad()
        outputs, dec_self_attns = model(dec_inputs_batch)
        # print("out:", outputs.view(-1, outputs.size(-1)).shape)
        # print("dec_out:", dec_outputs_batch.view(-1).shape)
        loss = criterion(outputs.view(-1, outputs.size(-1)), dec_outputs_batch.view(-1))
        epoch_loss += loss.item()

        print(f'Iter {it:4d} Loss: {loss:.4f} | Iter PPL: {math.exp(loss):7.4f}', file=ff)
        print(f'Iter {it:4d} Loss: {loss:.4f} | Iter PPL: {math.exp(loss):7.4f}')

        loss.backward()
        optimizer.step()

    return epoch_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for it, (val_enc_inputs_batch, val_enc_attn_mask_batch, val_enc_outputs_batch, val_dec_inputs_batch,
                 val_dec_outputs_batch, val_pocket_batch) in enumerate(loader):
            val_enc_inputs_batch, val_enc_attn_mask_batch = val_enc_inputs_batch.cuda(), val_enc_attn_mask_batch.cuda()
            val_enc_outputs_batch = val_enc_outputs_batch.cuda()
            val_dec_inputs_batch, val_dec_outputs_batch = val_dec_inputs_batch.cuda(), val_dec_outputs_batch.cuda()
            val_pocket_batch = val_pocket_batch.cuda()

            val_outputs, val_val_dec_self_attns = model(val_dec_inputs_batch)
            loss = criterion(val_outputs.view(-1, val_outputs.size(-1)), val_dec_outputs_batch.view(-1))
            epoch_loss += loss.item()

    return epoch_loss / len(loader)

'''Process dataset'''
print('---------- Process dataset')
data=pd.read_csv(config["datasets"], header=None)
data_list=data.iloc[:, config["datasets_col"]].tolist()
for index in range(len(data_list)):
    for j in range(config["tgt_len"]-len(data_list[index])):
        data_list[index]=data_list[index]+'/'
dec_outputs = []
dec_inputs=[]
for seq in data_list:
    dec_inputs.append([tokens.index(s) for s in seq])
    dec_outputs.append([tokens.index(s) for s in seq[1:]])
for i in dec_outputs:
    i.append(0)
print("dec_inputs:", dec_inputs[0])
print("dec_outputs:", dec_outputs[0])

loader = Data.DataLoader(MyDataSet(torch.tensor(dec_inputs), torch.tensor(dec_outputs)),
                         batch_size=config["batch_size"], shuffle=True)

'''Initialize model'''
print('---------- Initialize model')
transformer_model = Generator.Transformer(config_Generator).cuda()
# transformer_model = torch.load('model_50.pth') //load checkpoint model

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer_model.parameters(), lr=2e-4, betas=(0.9, 0.98), eps=1e-09,weight_decay=2e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.92)

print("batch_size:", config["batch_size"])
print("d_embed:", config["d_embed"])
print("d_ff:", config["d_ff"])
print("d_k=d_v:", config["d_k"])
print("n_layers:", config["n_layers"])
print("n_heads:", config["n_heads"])
print("optimizer & lr: Adam , 2e-3")

ff = open(config["save_addr"] + '/train.txt', 'w')
ff.write("d_embed: " + str(config["d_embed"]) + "\n")
ff.write("d_ff: " + str(config["d_ff"]) + "\n")
ff.write("d_k: " + str(config["d_k"]) + "\n")
ff.write("d_v: " + str(config["d_v"]) + "\n")
ff.write("n_layers:" + str(config["n_layers"]) + "\n")
ff.write("n_heads:" + str(config["n_heads"]) + "\n")
ff.write("batch_size:" + str(config["batch_size"]) + "\n")
ff.write("optimizer: " + "Adam" + "\n")

loss_epoch = []
best_valid_loss = 10
model_best = transformer_model
epoch_best = 0
print('---------- Training model')

for epoch in range(1, config["epochs"] + 1):
    start_time = time.time()
    train_loss = train(transformer_model, loader, optimizer, criterion)

    scheduler.step()
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    loss_epoch.append(train_loss)
    '''save model'''
    if epoch % 5 == 0:
        if torch.cuda.device_count() > 1:
            transformer_model.save_model(config["save_addr"] + '/model_' + str(epoch) + '.pth')
            #torch.save(transformer_model.module, config["save_addr"] + '/model_' + str(epoch) + '.pth')
        else:
            #torch.save(transformer_model, config["save_addr"] + '/model_' + str(epoch) + '.pth')
            transformer_model.save_model(config["save_addr"] + '/model_' + str(epoch) + '.pth')
    print(f'Epoch: {epoch:04} | Time: {epoch_mins}m {epoch_secs}s', file=ff)
    print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):7.4f}', file=ff)
    print(file=ff)
    print(f'Epoch: {epoch:04} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):7.4f}')

ff.close()


print('---------- Save result')
x = range(1, len(loss_epoch) + 1)
plt.plot(x, loss_epoch, 'b')
plt.legend(["train"])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(config["save_addr"] + '/loss.jpg')
