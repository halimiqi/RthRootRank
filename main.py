import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataloader
import config
import r_Model
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

def train(train_loader, model, loss, optimizer,epoch):


    return

def main():
    train_set = dataloader.EGG_Dataset(path = "data/eeg-eye-state_csv.csv")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKER)
    model =
    train(train_loader)
    return

if __name__ == "__main__":
    main()
