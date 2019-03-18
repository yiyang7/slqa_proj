import torch
import random
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np


class Classifier(nn.Module):
    def __init__(self, p_size, q_size, drop_prob=0.):
        super(Classifier, self).__init__()

        self.drop_prob = drop_prob
        self.sigmoid = nn.Sigmoid()

        self.linear1 = nn.Linear(p_size + q_size, p_size + q_size)
        self.linear2 = nn.Linear(p_size + q_size, p_size + q_size)
        
        self.dropout1 = nn.Dropout(p = 0.3)
        self.dropout2 = nn.Dropout(p = 0.3)

        self.proj_out = nn.Linear(p_size + q_size, 1)
        
        
    def forward(self, p_self_attn, q_self_attn, mask=None):
        
        batch_size = p_self_attn.shape[0]
        
        x = torch.cat([p_self_attn, q_self_attn], 2)
        
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_prob, self.training)

        x = self.linear2(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_prob, self.training)

        x = self.proj_out(x)
        pred_scores = self.sigmoid(x.squeeze(2))
        
        return pred_scores
        