# -*- coding:utf-8 -*-
import os
import random
import math
import copy
from tqdm import trange
import numpy as np
import predict_model
import torch
import torch.nn as nn
import torch.optim as optim
import re



class PolicyGradient(object):
    """ policy gradient"""
    def __init__(self, gen_loader, beta=0, score_up=1.2, score_down=0.6):
        self.data_loader = gen_loader
        self.beta = beta
        self.protbert = predict_model.Protbert()
        self.MLP_pre = predict_model.MLP()
        self.score_up = score_up
        self.score_down = score_down

    def get_reward(self, x, use_cuda, discriminator=None, use_prot=True):
        seq_len = len(x)
        if seq_len == 2 :
            rewards = np.array([-1])
        else:
            rewards = [0]*(seq_len-1)
            if x[1:].find(x[0]) != -1:
                pos = []
                for idx, char in enumerate(x):
                    if idx != 0 and char == x[0]:
                        pos.append(idx)
                for idx in pos:
                    rewards[idx - 1] = -1
            else:
                if x.find("/") == -1:
                    x_temp = x[1:]
                else:
                    x_temp = x[1:-1]
                if discriminator == None:
                    reward = 1
                else:
                    reward = discriminator.classify(self.data_loader.char_tensor(x_temp))- self.beta * abs(discriminator.classify(self.data_loader.char_tensor(x_temp)))
                seqs = []
                seqs.append(x_temp)
                if x_temp == []:
                    print(x + '\n')
                fea = self.protbert.pre(seqs)
                fea = np.array(fea)
                reward_stability = self.MLP_pre.predict(fea)
                if use_prot == True:
                    if reward_stability >= 1:
                        reward *= self.score_up
                    else:
                        reward *= self.score_down
                for i in range(len(rewards)):
                    rewards[i] = reward
        if use_cuda :
            return torch.Tensor(rewards).cuda()
        else :
            return torch.Tensor(rewards)