import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def my_loss(output, similiar_target,unsim_list,num_of_unsimiliar = 50):
    sig = nn.Sigmoid()
    V_qi = get_V(output, similiar_target)
    V_qj_list = []
    for i in range(num_of_unsimiliar):
        with torch.no_grad():
            V_qj = get_V(output, unsim_list[i])
            sig(V_qi - V_qj)




    return

def get_V(output, target):
    V = torch.norm((output - target), p=1)
    return V