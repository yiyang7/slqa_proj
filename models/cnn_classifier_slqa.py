import torch
import random
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from allennlp.modules.input_variational_dropout import InputVariationalDropout

class CNN_Classifier(nn.Module):
    def __init__(self, in_channels, out_channels, window_size, emb_len, drop_prob=0.):
        super(CNN_Classifier, self).__init__()
        
        # cnn
        kernel_size = [window_size,emb_len]
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size)
        # pooling
        self.pool2d = nn.MaxPool2d(kernel_size=[emb_len-window_size+1,1])
        # linear
#         self.linear1 = nn.Linear(out_channels, 1)
        self.linear = nn.Linear(out_channels, 1)
        # dropout
        self.drop_prob = drop_prob
        # sigmoid
        self.sigmoid = nn.Sigmoid()
        
#         self.dropout = nn.Dropout(dropout_p)
#         self.batch_norm = nn.BatchNorm1d(1)
        
        
    def forward(self, p_prime, q_prime, mask=None):
        
#         print("p_prime: ", p_prime.shape) # (#,seq_len,hidden_size)
#         print("q_prime: ", q_prime.shape)
        batch_size, doc_len, emb_len = p_prime.shape
        _, query_len, _ = q_prime.shape
        
        x = torch.cat([p_prime, q_prime], 1).view(batch_size, 1, doc_len+query_len, emb_len)
#         print("x: ", x.shape)
        
        # conv layer
        conv_out = self.conv2d(x)
#         print("conv_out: ", conv_out.shape)
        
        # pooling layer
        pooling_out = self.pool2d(conv_out).squeeze()
#         print("pooling_out: ", pooling_out.shape) # [64, 32, 1, 1]

        # dropout
        pooling_out = F.dropout(pooling_out, self.drop_prob, self.training)
    
        # fully connected layer
        fc_out = self.linear(pooling_out)
#         print("fc_out: ", fc_out.shape)
        
        pred_scores = self.sigmoid(fc_out)
#         print("pred_scores: ", pred_scores.shape)
        
        return pred_scores

if __name__ == "__main__":
    drop_prob = 0.4
    batch_size = 64
    in_channels = 1
    out_channels = 32
    doc_len = 100
    query_len = 10
    emb_len = 128
    
    # init model
    seq_len = doc_len + query_len
    cnn = CNN_Classifier(in_channels, out_channels, emb_len, drop_prob=0.4)
    
    # input
    p_prime = torch.randn(batch_size, doc_len, emb_len)
    q_prime = torch.randn(batch_size, query_len, emb_len)
    
    out = cnn(p_prime, q_prime)
    
    
    