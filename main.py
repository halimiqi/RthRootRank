import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import dataloader
import config
import r_Model
from r_CNN import r_CNN
from r_LSTM import r_LSTM
def my_loss(output, similiar_target,unsim_list,num_of_unsimiliar = 50, r = 2, my_lambda = 0.1, M = 100000):
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
    V = torch.norm((output - target), p=1)
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
        for sample in batch_sample:
        # get X_q
            input = sample["seq"]
            label = sample["label"]
            output = model(input)
            # get X_i
            X_i = train_set.generate_target_sample(label)
            loss = loss()
    return

def main():
    train_set = dataloader.EGG_Dataset(path = "data/eeg-eye-state_csv.csv")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKER)
    model = r_Model.r_Model(r_CNN,r_LSTM,lstm_input_feature = 14)
    optimizer = torch.optim.Adam(model.parameters(), config.LR, betas = (config.ADAM_BETA1, config.ADAM_BETA2))
    train(train_set,train_loader, model, my_loss,optimizer)
    return

if __name__ == "__main__":
    main()
