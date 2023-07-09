import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class Config():
    def __init__(self, pro_vocab_size, use_cuda, tgt_len, d_embed, d_ff, d_k, d_v, n_layers, n_heads):
        self.use_cuda = use_cuda
        self.pro_vocab_size = pro_vocab_size
        self.tgt_len = tgt_len
        self.d_embed = d_embed
        self.d_ff = d_ff
        self.d_k = d_k
        self.d_v = d_v
        self.n_layers = n_layers
        self.n_heads = n_heads


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.config = config

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.config.d_k ** 0.5) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)       # attn = self.dropout(F.softmax(attn, dim=-1))
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        
        self.config = config

        self.W_Q = nn.Linear(config.d_embed, config.d_k * config.n_heads, bias=False)
        self.W_K = nn.Linear(config.d_embed, config.d_k * config.n_heads, bias=False)
        self.W_V = nn.Linear(config.d_embed, config.d_v * config.n_heads, bias=False)
        self.fc = nn.Linear(config.n_heads * config.d_v, config.d_embed, bias=False)
        self.ln = nn.LayerNorm(config.d_embed)
        
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_embed]
        input_K: [batch_size, len_k, d_embed]
        input_V: [batch_size, len_v(=len_k), d_embed]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.config.n_heads, self.config.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.config)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.config.n_heads * self.config.d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_embed]
        return self.ln(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_embed, bias=False)
        )
        self.ln = nn.LayerNorm(config.d_embed)
        
    def forward(self, inputs):
        # inputs: [batch_size, seq_len, d_embed]

        residual = inputs
        output = self.fc(inputs)
        return self.ln(output + residual) # [batch_size, seq_len, d_embed]


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(config)
        # self.dec_enc_attn = MultiHeadAttention(config)
        self.pos_ffn = PoswiseFeedForwardNet(config)

    def forward(self, dec_inputs,dec_self_attn_mask):

        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_embed]
        return dec_outputs, dec_self_attn



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.config = config

        self.tgt_emb = nn.Embedding(config.pro_vocab_size, config.d_embed)
        # self.tgt_emb = OnehotEncode()
        self.pos_emb = PositionalEncoding(config.d_embed, max_len=config.tgt_len)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])

    def get_attn_pad_mask(self, seq_q, seq_k):
        '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
        '''
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
    
    def get_attn_subsequence_mask(self, seq):
        '''
        seq: [batch_size, tgt_len]
        batch_size, len_q = seq.size()
        subsequence_mask = seq.unsqueeze(1).repeat(1, len_q, 1)
        subsequence_mask[:, :, :] = 1
        subsequence_mask = torch.triu(subsequence_mask, 1)
        '''
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        
        return subsequence_mask.cuda()
    
    def forward(self, dec_inputs):

        dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_embed]

        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, tgt_len, d_embed]
        dec_self_attn_pad_mask = self.get_attn_pad_mask(dec_inputs, dec_inputs) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = self.get_attn_subsequence_mask(dec_inputs)  # [batch_size, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0) # [batch_size, tgt_len, tgt_len]


        dec_self_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_embed], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        return dec_outputs, dec_self_attns


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        # self.encoder = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        # self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.projection = nn.Linear(config.d_embed, config.pro_vocab_size, bias=False)

        
    def forward(self,dec_inputs):

        # dec_outpus: [batch_size, tgt_len, d_embed], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attns = self.decoder(dec_inputs)
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, vocab_size]
        return dec_logits, dec_self_attns


    def generate(self,start_symbol,max_len,temperature=1):


        tokens = ['/', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S', 'T', 'C', 'P', 'U', 'O']
        self.eval()
        # 二级结构标签（字母），模型实例，生成条数，最大长度，温度因子（默认为1）
        # gens=generate_seq(start_symbol,model,20,max_len)

        # temperature_dec_input = temperature_decoder(model,tokens.index(start_symbol),max_len)

        dec_input = torch.zeros(1, max_len).type(dtype=torch.int64)
        dec_input=dec_input.cuda()
        next_symbol = tokens.index(start_symbol)
        with torch.no_grad():
            for i in range(0, max_len):
                dec_input[0][i] = next_symbol
                dec_output,_ = self.decoder(dec_input)
                projected = self.projection(dec_output)  # projected (1, tgt_len, vocab_size)

                prob = projected.squeeze(0)[i]
                next_word = torch.multinomial(F.softmax(prob/temperature,0), 1)

                next_symbol = next_word.item()

        predict, _ = self(dec_input)  # (1, tgt_len, vocab_size)
        predict = predict.squeeze()
        predict = predict.data.max(1, keepdim=False)[1]
        gen = start_symbol
        for j in predict:
            if 0 <= j.item() <= 4:
                break
            gen += tokens[j]
        gen += '/'
        if len(gen) > max_len:
            return gen[:max_len] + '/'
        else:
            return gen
        
    def get_prob(self,seq):
        tokens = ['/', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S', 'T', 'C', 'P', 'U', 'O']
        dec_inputs=[]
        dec_inputs.append([tokens.index(s) for s in seq])
        dec_outputs, dec_self_attns = self.decoder(torch.tensor(dec_inputs).cuda())
        dec_logits = self.projection(dec_outputs)
        dec_logits=F.log_softmax(dec_logits.squeeze(0),1)
        prob=[]
        
        for i in range(dec_logits.shape[0]-1):
            prob.append(dec_logits[i][tokens.index(seq[i+1])])

        prob = torch.tensor(prob).cuda()
        return prob

    def save_model(self, path):
        """
        Saves model parameters into the checkpoint file.

        Parameters
        ----------
        path: str
            path to the checkpoint file model will be saved to.
        """
        torch.save(self.state_dict(), path)








