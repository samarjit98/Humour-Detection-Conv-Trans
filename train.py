import os
import gc
import torch
import argparse
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from models import *
from dataset import *

parser = argparse.ArgumentParser(description='PyTorch Aggression Classification')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') 
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay') 
parser.add_argument('--batch_size', default=64, type=int) 
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--preparedata', type=int, default=1)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()

print('==> Creating networks..')
model = CNN(num_chars=67).to(device)
params = model.parameters()
optimizer = optim.Adam(params, lr=args.lr)

print('==> Loading data..')
trainset = HumorDataset()
testset = HumorDataset(test=True)

def train(currepoch, epoch):
    dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)
    print('\n=> Epoch: {}'.format(currepoch))
    
    train_loss, correct, total = 0, 0, 0

    for batch_idx in range(len(dataloader)):
        inputs, targets = next(dataloader)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        y_pred = model(inputs)

        loss = criterion(y_pred, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with open("./logs/model_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        with open("./logs/model_train_acc.log", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        del inputs
        del targets
        gc.collect()
        torch.cuda.empty_cache()
        torch.save(model.state_dict(), './weights_cnn_deeper/network_train.ckpt')
        with open("./information/network_info.txt", "w+") as f:
            f.write("{} {}".format(currepoch, batch_idx))
        print('Batch: [{}/{}], Loss: {}, Train Loss: {:.4f} , Acc: {:.4f} ({}/{})'\
                                                                            .format(batch_idx,\
                                                                                    len(dataloader),\
                                                                                    loss.item(),\
                                                                                    train_loss/(batch_idx+1),\
                                                                                    100.0*correct/total,\
                                                                                    correct, \
                                                                                    total), end='\r')

    torch.save(model.state_dict(), './checkpoints_cnn_deeper/network_train_epoch_{}.ckpt'.format(currepoch + 1))
    print('\n=> Classifier Network Train: Epoch [{}/{}], Loss:{:.4f}'\
                                                                .format(currepoch+1,\
                                                                        epoch,\
                                                                        train_loss/len(dataloader)))

def test(currepoch, epoch):
    dataloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)
    print('\n=> Indic Testing Epoch: {}'.format(currepoch))
    
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_idx in range(len(dataloader)):
            inputs, targets = next(dataloader)
            inputs, targets = inputs.to(device), targets.to(device)

            y_pred = model(inputs)

            loss = criterion(y_pred, targets)

            test_loss += loss.item()
            _, predicted = y_pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            with open("./logs/model_test_loss.log", "a+") as lfile:
                lfile.write("{}\n".format(test_loss / total))

            with open("./logs/model_test_acc.log", "a+") as afile:
                afile.write("{}\n".format(correct / total))

            del inputs
            del targets
            gc.collect()
            torch.cuda.empty_cache()
            print('Batch: [{}/{}], Loss: {}, Test Loss: {:.4f} , Acc: {:.4f} ({}/{})'\
                                                                                    .format(batch_idx,\
                                                                                        len(dataloader),\
                                                                                        loss.item(),\
                                                                                        test_loss/(batch_idx+1),\
                                                                                        100.0*correct/total,\
                                                                                        correct, \
                                                                                        total), end='\r')

    print('\n=> Classifier Network Test: Epoch [{}/{}], Loss:{:.4f}'.format(currepoch+1,\
                                                                                epoch,\
                                                                                test_loss/len(dataloader)))

print('==> Training starts..')
for epoch in range(args.epochs):
    train(epoch, args.epochs)
    test(epoch, args.epochs)