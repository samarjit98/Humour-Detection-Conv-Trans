import os
import gc
import torch
import argparse
import matplotlib
import numpy as np
from collections import Counter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

from dataset import *

import matplotlib.pyplot as plt
import matplotlib

indic = [
        'humourous',
        'non-humourous'
    ]

class CNN(nn.Module):

    def __init__(self, 
                num_chars=38, 
                embedding_dim=128, 
                labels=2):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.convnet1 = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.MaxPool1d(3),
                                nn.BatchNorm1d(embedding_dim),
                                nn.Dropout(),
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.MaxPool1d(3),
                                nn.BatchNorm1d(embedding_dim),
                                nn.Dropout(),
                            )
        self.attn1_1 = nn.TransformerEncoderLayer(128, 8, 
                                                dim_feedforward=512, 
                                                dropout=0.5, 
                                                activation='relu')
        self.attn1_2 = nn.TransformerEncoderLayer(128, 8, 
                                                dim_feedforward=512, 
                                                dropout=0.5, 
                                                activation='relu')
        self.convnet2 = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.MaxPool1d(3),
                                nn.BatchNorm1d(embedding_dim),
                                nn.Dropout(),
                            )
        self.attn2 = nn.TransformerEncoderLayer(128, 8, 
                                                dim_feedforward=512, 
                                                dropout=0.5, 
                                                activation='relu')
        self.convnet3 = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.MaxPool1d(3),
                                nn.BatchNorm1d(embedding_dim),
                                nn.Dropout(),
                            )
        self.label = nn.Linear(2*embedding_dim, labels)

    def forward(self, x):
        this_batch = x.shape[0]
        x = self.embedding(x)

        x = x.permute(0, 2, 1)
        x = self.convnet1(x)
        x = x.permute(0, 2, 1)

        x = x.permute(1, 0, 2)
        x = self.attn1_1(x)
        x = self.attn1_2(x)
        x = x.permute(1, 0, 2)

        x = x.permute(0, 2, 1)
        x = self.convnet2(x)
        x = x.permute(0, 2, 1)

        x = x.permute(1, 0, 2)
        x = self.attn2(x)
        x = x.permute(1, 0, 2)

        x = x.permute(0, 2, 1)
        x = self.convnet3(x)
        x = x.permute(0, 2, 1)
        x = x.reshape((this_batch, -1))
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load('./checkpoints_cnn/network_train_epoch_75.ckpt'))

testset = HumorDataset(test=True)
dataloader = DataLoader(testset, batch_size=1, shuffle=True)

nb_classes = 8

# Initialize the prediction and label lists(tensors)
predlist = torch.zeros(0,dtype=torch.long, device='cpu')
lbllist = torch.zeros(0,dtype=torch.long, device='cpu')

X = []
Y = []

with torch.no_grad():
    for i, (input, target) in enumerate(dataloader):
        input = input.to(device)
        output = model(input)
        output = output[0].detach().cpu().numpy()
        print(indic[int(target[0].numpy())])
        X.append(output)
        Y.append(indic[int(target[0].numpy())])

from yellowbrick.text import TSNEVisualizer

# Create the visualizer and draw the vectors
colors = 'r', 'b'
tsne = TSNEVisualizer(colors=colors)
tsne.fit(X, Y)
tsne.show(outpath='./plots/tsne_cnn.png')

