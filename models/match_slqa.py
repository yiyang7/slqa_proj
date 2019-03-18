import torch
import torch.nn as nn
import torch.nn.functional as F

from util import masked_softmax
from .utils_slqa import BilinearSeqAtt

"""
---------------------------------------------------------------
Match

---------------------------------------------------------------
"""        
class Match(nn.Module):
    def __init__(self, u_feat_size):
        super(Match, self).__init__()
        
        input_dim1 = u_feat_size*2 # 640
        input_dim2 = u_feat_size*4 # 1280
        self.bilinear1 = BilinearSeqAtt(input_dim1, input_dim2)
        self.bilinear2 = BilinearSeqAtt(input_dim1, input_dim2)

        
    def forward(self, d_double_prime, q_bold, mask):
#         print("d_double_prime: ", d_double_prime.shape) # [64, 100, 1280]
#         print("q_bold: ", q_bold.shape) # [64, 1, 640]
        
        p_start = self.bilinear1(q_bold, d_double_prime) # [64, 1, 640] * W.T * [64, 100, 1280]
#         print("p_start: ", p_start.shape) # [64, 1, 100]

        p_start = masked_softmax(p_start.squeeze(1), mask, dim=1, log_softmax = True)
#         print("p_start: ", p_start.shape) # [64, 100]
        
        p_end = self.bilinear2(q_bold, d_double_prime)
#         print("p_end: ", p_end.shape) # [64, 1, 100]

        p_end = masked_softmax(p_end.squeeze(1), mask, dim=1, log_softmax = True)
#         print("p_end: ", p_end.shape) # [64, 100]
        
        return p_start, p_end