import csv
import time
import math
import numpy as np
import warnings

from rdkit import Chem
from rdkit import DataStructs
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

def read_sequence_file(filename, unique=True, add_end_tokens=False):
    """
    Reads sequences from file. File must contain one sequence string per line
    with \n token in the end of the line.

    Args:
        filename (str): path to the file
        unique (bool): return only unique SMILES

    Returns:
        sequences (list): list of SMILES strings from specified file.
        success (bool): defines whether operation was successfully completed or not.

    If 'unique=True' this list contains only unique copies.
    """
    f = open(filename, 'r')
    sequences = []
    for line in f:
        if add_end_tokens:
            sequences.append(line[:-1] + '/')
        else:
            sequences.append(line[:-1])
    if unique:
        sequences = list(set(sequences))
    else:
        sequences = list(sequences)
    f.close()
    return sequences, f.closed

def tokenize(smiles, tokens=None):
    """
    Returns list of unique tokens, token-2-index dictionary and number of
    unique tokens from the list of SMILES

    Parameters
    ----------
        smiles: list
            list of SMILES strings to tokenize.

        tokens: list, str (default None)
            list of unique tokens

    Returns
    -------
        tokens: list
            list of unique tokens/SMILES alphabet.

        token2idx: dict
            dictionary mapping token to its index.

        num_tokens: int
            number of unique tokens.
    """
    if tokens is None:
        tokens = list(set(''.join(smiles)))
        tokens = list(np.sort(tokens))
        tokens = ''.join(tokens)
    token2idx = dict((token, i) for i, token in enumerate(tokens))
    num_tokens = len(tokens)
    return tokens, token2idx, num_tokens

def read_object_property_file(path, delimiter=',', cols_to_read=[0, 1],
                              keep_header=False):
    f = open(path, 'r')
    reader = csv.reader(f, delimiter=delimiter)
    data_full = np.array(list(reader))
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data_full) > start_position
    data = [[] for _ in range(len(cols_to_read))]
    for i in range(len(cols_to_read)):
        col = cols_to_read[i]
        data[i] = data_full[start_position:, col]
    f.close()
    if len(cols_to_read) == 1:
        data = data[0]
    return data

