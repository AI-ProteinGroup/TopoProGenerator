# -*- coding: utf-8 -*-
import sys
sys.path.append('./src/')
import argparse
import json
import os
import torch
from data import GeneratorData
from src.generator_lstm import Generator

'''Loading config'''
parser = argparse.ArgumentParser(description="you should add those parameter for pretrain")
parser.add_argument("--config", dest="config", help="path to the JSON config file", required=True)
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)

if not os.path.exists(config['save_addr']):
    os.makedirs(config['save_addr'])
"""Loading data for the generator"""
tokens = ['/', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S', 'T', 'C', 'P', 'U', 'O']
gen_data = GeneratorData(training_data_path=config['datasets'], delimiter=',',
                         cols_to_read=[config['datasets_col']], keep_header=True, tokens=tokens)#这里的cols_to_read要注意

'''Setting up the generator'''
use_cuda  = torch.cuda.is_available()
optimizer_instance = torch.optim.Adam
generator = Generator(input_size=len(tokens), embed_size=config['embed_size_generator'], hidden_size=config['hidden_size_generator'],
                      output_size=len(tokens), n_layers=config['layers_generator'], use_cuda=use_cuda,
                      optimizer_instance=optimizer_instance, lr=config['learning_rate_generator'],
                      lr_warmup=config['learning_rate_generator'], epoch_warmup=0)

generator.pretrain(gen_data, config['epoch_pretrain'], config['save_addr'] + '/')


