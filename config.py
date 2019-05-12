
# CNN structure
CNN_EMBED_SIZE = 15  # CNN and LSTM embedding size should be same

# LSTM structure
LSTM_INPUT_EMBED_SIZE = 64
LSTM_HIDDEN_SIZE = 128
LSTM_OUTPUT_EMBED_SIZE = 15
# final Linear structure
LINEAR_EMBED_SIZE = 64

# trainging parameter
BATCH_SIZE = 32
NUM_WORKER = 4
NUMBER_DISSIMILAR = 20

LR = 0.1
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99

EPOCH = 100