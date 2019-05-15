import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config

class r_CNN(nn.Module):

    def __init__(self, input_channel,output_size, kernel_num_list = [16,32,64,64],stride_list = [1,2,2,1],norm_layer = None):
        super(r_CNN, self).__init__()
        self.output_size = output_size

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # the linear layer to convert size into embeddings
        self.conv1 = nn.Conv2d(input_channel, kernel_num_list[0], kernel_size = 3,stride=stride_list[0], padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(kernel_num_list[0], kernel_num_list[1], kernel_size = 3,stride=stride_list[1], padding=1, bias=False)
        self.conv3 = nn.Conv2d(kernel_num_list[1], kernel_num_list[2], kernel_size = 3,stride=stride_list[2], padding=1, bias=False)
        self.conv4 = nn.Conv2d(kernel_num_list[2], kernel_num_list[3], kernel_size = 3,stride=stride_list[3], padding=1, bias=False)
        self.fc5 = nn.Linear(512,256)
        self.fc6 = nn.Linear(256, output_size)

    def forward(self, image_x):
        x = self.conv1(image_x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = x.view(config.BATCH_SIZE, -1)
        x = self.fc5(x)
        cnn_output = self.fc6(x)
        #tag_scores = F.log_softmax(tag_space, dim=1)
        return cnn_output
