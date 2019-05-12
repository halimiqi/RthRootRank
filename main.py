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
from r_CNN import r_CNN
from r_LSTM import r_LSTM
def my_loss(output, similiar_target,unsim_list,num_of_unsimiliar = 50, r = 2, my_lambda = 0.1, M = 100000):
    """
    This is based on the base version
    :param output:
    :param similiar_target:
    :param unsim_list:
    :param num_of_unsimiliar:
    :param r:
    :param my_lambda:
    :param M:
    :return:
    """

    sig = nn.Sigmoid()
    loss = torch.tensor(0)
    V_qi = get_V(output, similiar_target)
    V_qj_list = []
    for i in range(num_of_unsimiliar):
        with torch.no_grad():
            V_qj = get_V(output, unsim_list[i])
            loss = loss + sig(V_qi - V_qj)
        loss = torch.pow((M/num_of_unsimiliar) * loss, 1/r)

    return loss

def get_V(output, target):
    V = torch.norm((output - target), p=1, dim = 1)
    return V

def train(train_set,train_loader, model, loss, optimizer):
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
        outputs = model(inputs)
        # get X_i
        X_i_list = []
        X_j_list_outer = []
        for label in labels:
            X_i = train_set.generate_target_sample(labels)
            X_i = model(X_i)
            X_i_list.apend(X_i)

        # get the negative label
            X_j_list = train_set.generate_negative_sample(config.NUMBER_DISSIMILAR,label)
            X_j_list = np.array(X_j_list)
            X_j_list = model(X_j_list)
            X_j_list_outer.append(X_j_list)
        loss = loss(outputs, X_i_list, X_j_list_outer,num_of_unsimiliar = len(X_j_list_outer)) ## the loss of the batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return

def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main():
    train_set = dataloader.EGG_Dataset(path = "data/eeg-eye-state_csv.csv")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKER)
    model = r_Model.r_Model(r_CNN,r_LSTM,cnn_input_channel = 1,lstm_input_feature = 14)
    optimizer = torch.optim.Adam(model.parameters(), config.LR, betas = (config.ADAM_BETA1, config.ADAM_BETA2))
    for i in range(config.EPOCH):
        train(train_set,train_loader, model, my_loss,optimizer)

        save_checkpoint({
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best = False )
    return

if __name__ == "__main__":
    main()
