import sys
sys.path.append('./src/')
import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
from sklearn import preprocessing
import src.generator_transformer as Generator
import argparse
import json

use_cuda = torch.cuda.is_available()

tokens = ['/', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S', 'T', 'C', 'P', 'U', 'O']
'''Load config'''
parser = argparse.ArgumentParser(description="you should add those parameter for pretrain")
parser.add_argument("--config", dest="config", help="path to the JSON config file", required=True)
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)

'''Load model'''
config_Generator = Generator.Config(pro_vocab_size=len(tokens), use_cuda=use_cuda, tgt_len=config["tgt_len"],
                         d_embed=config["d_embed"], d_ff=config["d_ff"], d_k=config["d_k"], d_v=config["d_v"], n_layers=config["n_layers"], n_heads=config["n_heads"])
model = Generator.Transformer(config=config_Generator)
model.load_state_dict(torch.load(config["generator_model"]))
if use_cuda:
    model = model.cuda()

'''Generate sequence'''
with open(config["seq_save"], 'a+') as f_w:
    num = 0
    while num <= config["num_seq"]:
        seq = model.generate(config["prime_str"], config["tgt_len"], config["temperature"])
        if len(seq) <= config["min_length"] + 2:
            continue
        f_w.write('>sequence' + str(num) + '\n')
        if len(seq) > config["max_length"] + 2:
            f_w.write(seq[1:config["max_length"] + 1] + '\n')
        else:
            f_w.write(seq[1:-1] + '\n')
        num += 1
    f_w.close()
