
# CNN structure
CNN_EMBED_SIZE = 15  # CNN and LSTM embedding size should be same


# LSTM structure
LSTM_INPUT_EMBED_SIZE = 64 # the input feature size of lstm
LSTM_HIDDEN_SIZE = 128 # the output feature size of lstm
LSTM_OUTPUT_EMBED_SIZE = 15  # the final embedding size after linear layer
# final Linear structure
LINEAR_EMBED_SIZE = 64
TIMESTAPE = 5
FEATURE_NUM = 14

# trainging parameter
BATCH_SIZE = 64
NUM_WORKER = 4
NUMBER_DISSIMILAR = 64

LR = 0.0001
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99
ADAM_LAMBDA = 0.00001

EPOCH = 100
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1