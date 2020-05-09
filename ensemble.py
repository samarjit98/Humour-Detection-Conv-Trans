import os
import gc
import torch
import argparse
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from models import *
from dataset import *
from statistics import mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('==> Creating networks..')
model1 = AttentionModel().to(device)
model2 = AttentionModel2().to(device)
model3 = CNN().to(device)

model1.load_state_dict(torch.load('./checkpoints_lstm1/network_train_epoch_104.ckpt'))
model2.load_state_dict(torch.load('./checkpoints_lstm2/network_train_epoch_45.ckpt'))
model3.load_state_dict(torch.load('./checkpoints_cnn/network_train_epoch_75.ckpt'))

print('==> Loading data..')
testset = HumorDataset(test=True)

def test():
    dataloader = DataLoader(testset, batch_size=1, shuffle=True)
    dataloader = iter(dataloader)
    
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        model1.eval()
        model2.eval()
        model3.eval()
        for batch_idx in range(len(dataloader)):
            inputs, targets = next(dataloader)
            inputs, targets = inputs.to(device), targets.to(device)


            y_pred1 = F.softmax(model1(inputs))
            y_pred2 = F.softmax(model2(inputs))
            y_pred3 = F.softmax(model3(inputs))

            _, predicted1 = y_pred1.max(1)
            _, predicted2 = y_pred2.max(1)
            _, predicted3 = y_pred3.max(1)
            total += targets.size(0)
            target = targets.cpu().numpy()[0]
            predicted1 = predicted1.cpu().numpy()[0]
            predicted2 = predicted2.cpu().numpy()[0]
            predicted3 = predicted3.cpu().numpy()[0]

            if(mode([predicted1, predicted2, predicted3]) == target):
                correct += 1

            del inputs
            del targets
            gc.collect()
            torch.cuda.empty_cache()
            print('Batch: [{}/{}], Acc: {:.4f} ({}/{})'.format(batch_idx,\
                                                                len(dataloader),\
                                                                100.0*correct/total,\
                                                                correct, \
                                                                total))


if __name__ == '__main__':
    test()
