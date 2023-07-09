# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('./src/')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import math
import random
import numpy as np
from tqdm import tqdm, trange
import pickle
from src.reinforcement_pean import PolicyGradient
from utils import read_sequence_file
from data import GeneratorData, DiscriminatorData
from src.discriminator import Discriminator
from src.generator_lstm import Generator
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json

def cosine_annealing_with_warm_restarts(epoch, lr_init, cycle_length, restart_lr_multiplier=0.95):
    """
    Implements the cosine annealing with warm restarts learning rate schedule.

    Args:
    - epoch: the current epoch
    - lr_init: the initial learning rate
    - cycle_length: the length of the first cycle, in epochs (default: 10)
    - restart_lr_multiplier: the factor by which the learning rate is multiplied after each restart (default: 1.0)
    - num_cycles: the number of cycles before the learning rate is no longer decayed (default: 3)

    Returns:
    - the learning rate for the current epoch
    """
    curr_cycle = epoch // cycle_length
    curr_cycle_epoch = epoch % cycle_length
    T_i = cycle_length * restart_lr_multiplier**curr_cycle
    lr_min = 0
    lr_max = lr_init
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(curr_cycle_epoch / T_i * math.pi))
    return lr

'''Load config'''
parser = argparse.ArgumentParser(description="you should add those parameter for pretrain")
parser.add_argument("--config", dest="config", help="path to the JSON config file", required=True)
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)

if not os.path.exists(config['save_addr']):
    os.makedirs(config['save_addr'])

tokens = ['/', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S', 'T', 'C', 'P', 'U', 'O']

'''Set up the generator'''
use_cuda = torch.cuda.is_available()

gen_data = GeneratorData(training_data_path=config["fine_tuning_datasets"], delimiter=',',
                         cols_to_read=[config["fine_tuning_datasets_col"]], keep_header=True, tokens=tokens)
generator = Generator(input_size=len(tokens), embed_size=config['embed_size_generator'], hidden_size=config['hidden_size_generator'],
                      output_size=len(tokens), n_layers=config['layers_generator'], use_cuda=use_cuda,
                      optimizer_instance=torch.optim.Adam, lr=config['learning_rate_generator'],
                      lr_warmup=config['pre_learning_rate_generator'], epoch_warmup=5)

generator.scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(generator.optimizer, T_0=5, T_mult=2)
generator.load_model(config["generator_model"], map_location='cuda')

'''Initialize the discriminator'''
discriminator = Discriminator(input_size=len(tokens), embed_size=config["embed_size_discriminator"], hidden_size=config["hidden_size_discriminator"],
                                    use_cuda=use_cuda, dropout=config["dropout_discriminator"], lr=config["learning_rate_discriminator"],
                                    optimizer_instance=torch.optim.Adam)
'''Read truth data for discriminator'''
truth_data, _ = read_sequence_file(config["truth_seq_datasets"])

'''Use Generator to generate some fake data for discriminator pretraining'''
fake_data = []
num = 0
fake_data_num = config["fake_data_num"]
print('---------- Using generator to generate fake data for discriminiator pretraining')
with torch.no_grad():
    while(num < config["fake_data_num"]):
        sample = generator.generate(gen_data,prime_str=config["prime_str"])
        if len(sample) <= 2:
            continue
        else:
            if sample[-1] == '/':
                fake_data.append(sample[1:-1])
            else:
                fake_data.append(sample[1:])
            num = num + 1

'''Define DiscriminatorData'''
print('---------- Loading DiscriminatorData')
random.shuffle(fake_data)
dis_loader = DiscriminatorData(truth_data=truth_data, fake_data=fake_data[0:len(truth_data)], tokens=tokens,
                                batch_size=64)

'''Pretrain Discriminator'''
print('---------- Pretrain Discriminator ...')
discriminator.train()
loss = discriminator.train_epochs(dis_loader, config["d_epoch"])

'---------------Adversarial Training--------------------'
policy = PolicyGradient(gen_data, score_up=config["predictor_score_up"], score_down=config["predictor_score_down"])
g_loss_all = []
g_loss_one_epoch = 0
lr = []

for epoch in range(config["num_epochs"]):
    g_loss_one_epoch = 0
    discriminator.eval()
    generator.optimizer.zero_grad()
    generator.train()
    for i in range(config["g_epoch"]):
        with torch.no_grad():
            sample = generator.generate(gen_data,prime_str=config["prime_str"])
            rewards = policy.get_reward(x=sample,use_cuda=use_cuda,discriminator=discriminator, use_prot=False)
        prob = generator.get_prob(sample).cuda()
        g_loss = (- torch.sum(prob*rewards)).requires_grad_(True)
        g_loss_one_epoch += g_loss.item()
        generator.optimizer.zero_grad()
        g_loss.backward()
        generator.optimizer.step()
    g_loss_all.append(g_loss_one_epoch/config["g_epoch"])
    generator.scheduler.step()
    generator.save_model(config['save_addr'] + '/generator_traintime' + str(epoch) + '.pt')

    '''Train the discriminator '''
    '''generate fake data for discriminator'''
    print('---------- Generating fake data for discriminator ...')
    fake_data = []
    num = 0
    generator.eval()
    with torch.no_grad():
        with open(config['save_addr'] + '/traintime' + str(epoch) + '.fasta', 'a+') as f_w:
            while(num < config["fake_data_num"]):
                sample = generate(gen_data,prime_str=config["prime_str"])
                if len(sample) == 2:
                    continue
                else:
                    if sample[-1] == '/':
                        fake_data.append(sample[1:-1])
                        f_w.write('>seq' + str(num) + '\n')
                        f_w.write(sample[1:-1] + '\n')
                    else :
                        f_w.write('>seq' + str(num) + '\n')
                        f_w.write(sample[0:] + '\n')
                        fake_data.append(sample[1:])
                    num = num + 1
            f_w.close()
    print('---------- Train Discriminator ...')
    random.shuffle(fake_data)
    dis_loader.update(truth_data = None, fake_data = fake_data[0:len(truth_data)])
    discriminator.train()
    loss = discriminator.train_epochs(dis_loader, config["d_epoch"])
    discriminator.save_model(config['save_addr'] + '/discriminator_traintime' + str(epoch) + '.pt')

print("---------- Train over")
plt.plot(g_loss_all)
plt.xlabel('Training iteration')
plt.ylabel('loss')
plt.savefig(config['save_addr'] + '/g_loss.png')
plt.close()





