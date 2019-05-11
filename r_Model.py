import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import r_CNN as rcnn
import r_LSTM as rlstm
import config
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch.nn.modules.module import Module


class my_Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, \text{in\_features})` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, \text{out\_features})` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=False):
        super(my_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class r_Model(nn.Module):

    def __init__(self, cnn,lstm,cnn_input_channel,lstm_input_feature, norm_layer = None):
        super(r_Model, self).__init__()
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # the linear layer to convert size into embeddings
        self.cnn = cnn(cnn_input_channel,config.CNN_EMBED_SIZE, kernel_num_list = [16,32,64,64],stride_list = [1,2,2,1],norm_layer = None)
        self.lstm = lstm(lstm_input_feature, config.LSTM_INPUT_EMBED_SIZE, config.LSTM_HIDDEN_SIZE, config.LSTM_OUTPUT_EMBED_SIZE)
        self.linear_weight = Parameter(torch.Tensor((config.LSTM_OUTPUT_EMBED_SIZE + config.CNN_EMBED_SIZE), config.LINEAR_EMBED_SIZE))
    def forward(self, x):
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        # x = self.relu(x)
        # x = self.conv4(x)
        # x = self.relu(x)
        # x = self.fc5(x)
        # cnn_output = self.fc6(x)
        x = self.cnn(x)
        y = self.lstm(x)
        output = torch.cat((x, y), 2)
        bias = output.mean(2)
        output = output.sub(bias)
        output = F.linear(output, self.linear_weight,bias = None)
        output = F.tanh(output)
        #tag_scores = F.log_softmax(tag_space, dim=1)
        return output
