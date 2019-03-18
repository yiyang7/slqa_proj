import torch
import torch.nn as nn
import torch.nn.functional as F
from util import masked_softmax
from .utils_slqa import BilinearSeqAtt

# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .layers import RNNEncoder
"""
---------------------------------------------------------------
Self_Attention

---------------------------------------------------------------
"""  
class Self_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, drop_prob=0.):
        super(Self_Attention, self).__init__()
                
        self.bilstm1 = RNNEncoder(input_size=input_size, hidden_size=hidden_size, drop_prob=drop_prob, num_layers = 2)
        self.D_dim = hidden_size*2
        self.q_prime_dim = input_size
        self.bilinearAtt = BilinearSeqAtt(self.D_dim, self.D_dim)
        
        # fuse
        self.linear1 = nn.Linear(self.D_dim*4, self.D_dim, bias=True)
        self.tanh = nn.Tanh()
        
        self.bilstm2 = RNNEncoder(input_size=self.D_dim, hidden_size=self.D_dim, drop_prob=drop_prob, num_layers = 2)
        self.bilstm3 = RNNEncoder(input_size=self.q_prime_dim, hidden_size=self.q_prime_dim, num_layers = 2, drop_prob=drop_prob)
    
        # gamma
        self.linear2 = nn.Linear(self.q_prime_dim*2, 1, bias=False)
       
        # gamma for p
        self.linear3 = nn.Linear(self.D_dim, 1, bias=False)

        
    def forward(self, p_prime, q_prime, doc_len, query_len, doc_mask, query_mask):
        
        batch_size, _, emb_size = p_prime.size()

        d = self.bilstm1(p_prime, doc_len)
        l = self.bilinearAtt(d,d) 
        l = masked_softmax(l, doc_mask.unsqueeze(1), dim=2)
        
        d_tilde = torch.bmm(l,d)
        

        # FUSE
        concat_input_fuse = torch.cat((d,d_tilde,d*d_tilde,d-d_tilde),2)
        d_prime = self.tanh(self.linear1(concat_input_fuse))


        d_double_prime = self.bilstm2(d_prime, doc_len)
        
        # Get q_double_prime
        q_double_prime = self.bilstm3(q_prime, query_len)
        
        # Get gamma
        gamma = self.linear2(q_double_prime).permute(0,2,1)
        gamma = masked_softmax(gamma, query_mask.unsqueeze(1), dim=2)        
        q_bold = torch.bmm(gamma, q_double_prime)#.squeeze(1)

#==============================compute 1-dim self attention for P =====================        
        p_double_prime = d
        gamma_p = self.linear3(p_double_prime).permute(0,2,1)
        gamma_p = masked_softmax(gamma_p, doc_mask.unsqueeze(1), dim=2)
        p_bold = torch.bmm(gamma_p, p_double_prime)

#=======================================================================================        
        
        return d_double_prime, q_bold, p_bold