import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import shutil
import time
import dataloader
import config
import r_Model
import random
import logistic_regression as logreg
from r_CNN import r_CNN
from r_LSTM import r_LSTM
def my_loss(output, similiar_target,unsim_list,batch_num = 50, r = 2, my_lambda = 0.1, M = 10000):
    """
    This is based on the base version
    :param output:
    :param similiar_target:
    :param unsim_list:
    :param batch_num:
    :param r:
    :param my_lambda:
    :param M:
    :return:
    """
    sig = nn.Sigmoid()
    loss = 0
    V_qi = get_V(output, similiar_target)
    V_qj_list = []
    for i in range(batch_num):
        #with torch.no_grad():
        V_qj = get_V(output[i], unsim_list[i])
        #test = sum(sig(V_qi[i] - V_qj))
        loss = loss + torch.pow((M/unsim_list[i].shape[0])*sum(sig(V_qi[i] - V_qj)), 1/r)
        #loss = torch.pow((M/unsim_list[i].shape[0]) * loss, 1/r)
    return loss

def get_V(output, target):
    V = torch.norm((output - target), p=1, dim = 1)
    return V

def train(train_set,train_loader, model, loss_func, optimizer, label_count_dict):
    """
    train one epoch
    :param train_loader:
    :param model:
    :param loss:
    :param optimizer:
    :return:
    """
    # switch the model in train mode
    model.train()
    start = time.time()
    # read the data
    for i , batch_sample in enumerate(train_loader):
        # get X_q
        inputs = batch_sample["seq"]
        labels = batch_sample["label"]
        outputs = model(inputs.float())
        # get X_i
        X_i_list = []
        X_j_list_outer = []
        for label in labels:
            X_i = train_set.generate_target_sample(label)
            X_i_list.append(X_i)
        # get the negative label
            # this represent M in the paper, M is the total unsimiliar list
            M = 0
            for k in label_count_dict:
                if k != str(label):
                    M += label_count_dict[k]

            X_j_list = train_set.generate_negative_sample(config.NUMBER_DISSIMILAR,label)
            X_j_list = np.array(X_j_list)
            X_j_list = model(torch.Tensor(X_j_list))
            X_j_list_outer.append(X_j_list)
        X_i_list = model(torch.Tensor(X_i_list))
        loss = loss_func(outputs, X_i_list, X_j_list_outer,batch_num = len(X_j_list_outer), r = config.R, M = M) ## the loss of the batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss

def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_checkpoint(model,optimizer,filename = "model_best.pth.tar"):
    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer,epoch

def main():
    min_loss = 0
    is_best = False
    epoch = 0
    train_set = dataloader.EGG_Dataset(path = "data/eeg-eye-state_csv.csv")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKER)
    model = r_Model.r_Model(r_CNN,r_LSTM,cnn_input_channel = 1,lstm_input_feature = 14, cnn_width=config.TIMESTAPE, cnn_height=config.FEATURE_NUM)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), config.LR, betas = (config.ADAM_BETA1, config.ADAM_BETA2),weight_decay = config.ADAM_LAMBDA / 2)
    if config.RESUME:
        model, optimizer,epoch = load_checkpoint(model,optimizer,filename = "model_best.pth.tar")
    # get the label count dictionary: key is str(label) value is the count of every label in train set
    label_count_dict = train_set.Num_for_each_class()
    for i in range(epoch, config.EPOCH):
        print_loss = train(train_set,train_loader, model, my_loss,optimizer, label_count_dict)
        if min_loss == 0:
            min_loss = print_loss
        else:
            if min_loss > print_loss:
                min_loss = print_loss
                is_best = True
        print("epoch[%d]\tloss:%f"%(i,print_loss))
        if i % 3 == 0:
            train_list, train_label = train_set.get_train_data()
            train_x = model(train_list.float())
            logreg.LogisticRegression()
            return
        save_checkpoint({
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best = is_best )
        is_best = False
    return

if __name__ == "__main__":
    #main()
    train_set = dataloader.EGG_Dataset(path="data/eeg-eye-state_csv.csv")
    train_list, train_label = train_set.get_train_data()
    train_list = np.array(train_list)
    train_label = np.array(train_label)
    a = 1