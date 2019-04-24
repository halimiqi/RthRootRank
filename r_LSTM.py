# -*- coding: utf-8 -*-
# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2output = nn.Linear(hidden_dim, output_size)  # this can be viewed as full connect layer

    def forward(self, seq):
        embeds = self.fc1(seq)
        lstm_out, (last_h, last_c) = self.lstm(embeds.view(len(seq), 1, -1))
        r_lstm_out = self.hidden2output(last_h)
        #tag_scores = F.log_softmax(tag_space, dim=1)
        return r_lstm_out

# model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
# loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
#
# # See what the scores are before training
# # Note that element i,j of the output is the score for tag j for word i.
# # Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)
#
# for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
#     for sentence, tags in training_data:
#         # Step 1. Remember that Pytorch accumulates gradients.
#         # We need to clear them out before each instance
#         model.zero_grad()
#
#         # Step 2. Get our inputs ready for the network, that is, turn them into
#         # Tensors of word indices.
#         sentence_in = prepare_sequence(sentence, word_to_ix)
#         targets = prepare_sequence(tags, tag_to_ix)
#
#         # Step 3. Run our forward pass.
#         tag_scores = model(sentence_in)
#
#         # Step 4. Compute the loss, gradients, and update the parameters by
#         #  calling optimizer.step()
#         loss = loss_function(tag_scores, targets)
#         loss.backward()
#         optimizer.step()
#
# # See what the scores are after training
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#
#     print(tag_scores)

