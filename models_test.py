import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from losses import *
from utils import Update_mb
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
import losses
from torch.nn.modules.module import Module

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class model(torch.nn.Module):
    def __init__(self,max_seqlen,feature_size,Vitblock_num,cross_clip,split,beta,delta):
        super(model, self).__init__()
        self.feature_size = feature_size
        self.scorer = Scorer(n_feature=feature_size)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True, is_test=False):
        if not is_training:
            scores = self.scorer(inputs,is_training)
            return scores

class Scorer(torch.nn.Module):
    def __init__(self, n_feature):
        super(Scorer, self).__init__()
        self.fc1 = nn.Linear(n_feature, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.classifier = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def __del__(self):
        print("model deleted")

    def forward(self, inputs, is_training=True):
        if is_training:
            _, B, C= inputs.size()
            inputs = inputs.reshape(-1,1,C)
            x = self.relu(self.fc1(inputs))  # 2048
            x = self.dropout(x)
            x = self.relu(self.fc2(x))  # 2048
            x = self.dropout(x)
            x = self.classifier(x)
            score = self.sigmoid(x)
            return score.reshape(-1,B,1)
        else:
            x = self.relu(self.fc1(inputs))  # 2048
            x = self.relu(self.fc2(x))  # 2048
            x = self.classifier(x)
            score = self.sigmoid(x)
            return score