import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # TensorFlow occupies 20% of GPU, which can be reduced
session = tf.compat.v1.Session(config=config)
from tensorflow import keras
from tensorflow.keras import layers
import torch
from transformers import BertModel, BertTokenizer
import datetime
import re
import os
import requests
from tqdm.auto import tqdm

class Protbert():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('../transformer/protbert', do_lower_case=False )
        self.model = BertModel.from_pretrained('../transformer/protbert')
        self.model = self.model.cuda()
        self.model = self.model.eval()

    def pre(self, seqs):
        res = []
        for seq in seqs:
            seq_pre = ''
            for i in range(len(seq)):
                seq_pre += seq[i]
                if i < len(seq) - 1:
                    seq_pre += ' '
            res.append(seq_pre)
        sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in res]
        ids = self.tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).cuda()
        attention_mask = torch.tensor(ids['attention_mask']).cuda()
        with torch.no_grad():
            embedding = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        embedding = embedding.cpu().numpy()
        features = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len - 1]
            features.append(seq_emd[-1])
        return features


class MLP():
    def __init__(self):
        self.model = keras.models.load_model('predict_model.h5')

    def predict(self, feas):
        return self.model.predict(feas)
