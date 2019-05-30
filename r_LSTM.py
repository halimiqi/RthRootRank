# -*- coding: utf-8 -*-
# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
torch.manual_seed(1)

# Prepare data:

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {} # the dictionary records the words and the index
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

######################################################################
# Create the model:


class r_LSTM(nn.Module):

    def __init__(self,input_size, embedding_dim, hidden_dim, output_size):
        super(r_LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # the linear layer to convert size into embeddings
        self.fc1 = nn.Linear(input_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2output = nn.Linear(hidden_dim, output_size)  # this can be viewed as full connect layer

    def forward(self, seq):
        embeds = self.fc1(seq)
        lstm_out, (last_h, last_c) = self.lstm(embeds)
        r_lstm_out = self.hidden2output(lstm_out[:,-1,:])
        #tag_scores = F.log_softmax(tag_space, dim=1)
        return r_lstm_out
