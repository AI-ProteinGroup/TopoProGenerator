# -*- coding: utf-8 -*-
import sys
sys.path.append('./src/')

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
import pickle
from data import GeneratorData
from src.generator_lstm import Generator
import argparse
import json

'''Loading config'''
parser = argparse.ArgumentParser(description="you should add those parameter for pretrain")
parser.add_argument("--config", dest="config", help="path to the JSON config file", required=True)
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)

'''Load model'''
use_cuda = torch.cuda.is_available()
tokens = ['/', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S', 'T', 'C', 'P', 'U', 'O']

gen_data = GeneratorData(training_data_path=config["dataset"], delimiter=',',
                         cols_to_read=[config["dataset_col"]], keep_header=True, tokens=tokens)

generator = Generator(input_size=len(tokens), embed_size=config['embed_size_generator'], hidden_size=config['hidden_size_generator'],
                      output_size=len(tokens), n_layers=config['layers_generator'], use_cuda=use_cuda,
                      optimizer_instance=torch.optim.Adam, lr=config['learning_rate_generator'],
                      lr_warmup=config['learning_rate_generator'], epoch_warmup=0)

generator.load_model(config["model"], map_location='cuda')

with open(config["seq_save"], 'a+') as f_w:
    num = 0
    while num < config["num_seq"]:
        seq = generator.generate(gen_data, temperature=config["temperature"])
        if len(seq) <= config["min_length"] + 2:
            continue
        f_w.write('>sequence' + str(num) + '\n')
        if len(seq) > config["max_length"] + 2:
            f_w.write(seq[1:config["max_length"] + 1] + '\n')
        else:
            f_w.write(seq[1:-1] + '\n')
        num += 1
    f_w.close()



