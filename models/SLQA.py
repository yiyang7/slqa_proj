"""
Implement SLQA network
"""
from __future__ import print_function

from . import layers
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import masked_softmax

from .encoder_slqa import Encoder
from .co_attention_slqa import Co_Attention
# from .self_attention_slqa import Self_Attention
from .self_attention_new import Self_Attention
from .match_slqa import Match
from .utils_slqa import BilinearSeqAtt 
from .classifier_slqa import Classifier
from .cnn_classifier_slqa import CNN_Classifier
from .answer_module_slqa import Answer_Module_slqa

import numpy as np

"""
---------------------------------------------------------------
SLQA

---------------------------------------------------------------
"""
class SLQA_no_char(nn.Module):
    
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(SLQA_no_char, self).__init__()
        
        self.emb_size = 300
        
        #encoder
        self.encoder = Encoder(word_vectors, hidden_size, drop_prob)
        # co_attention
        self.u_feat_size = hidden_size*2 + self.emb_size
        self.co_attention = Co_Attention(self.u_feat_size, drop_prob)
        # self_attention
        self.self_attention = Self_Attention(self.u_feat_size, self.u_feat_size, drop_prob)


        # CNN classifier
        in_channels = 1
        out_channels = 64
        window_size = 4
        self.cnn_classifier = CNN_Classifier(in_channels, out_channels, window_size, self.u_feat_size)

        # classifier
        self.classifier = Classifier(self.u_feat_size*2, self.u_feat_size*2)
        
        # match
        input_size = self.u_feat_size
        self.match = Match(input_size)

        # answer module
#         self.anwer_module = Answer_Module_slqa()

#         input_size = u_feat_size*2
#         input_dim1 = u_feat_size*2
#         input_dim2 = u_feat_size*4
#         self.bilinear1 = BilinearSeqAtt(input_dim1, input_dim2)
#         self.bilinear2 = BilinearSeqAtt(input_dim1, input_dim2)
         
#         self.classifier_new = Classifier_new(self.u_feat_size, self.u_feat_size)

    
    def forward(self, doc_w_idxs, query_w_idxs, doc_c_idxs, query_c_idxs, pred_cutoff = 0.3):    
        doc_mask = torch.zeros_like(doc_w_idxs) != doc_w_idxs
        query_mask = torch.zeros_like(query_w_idxs) != query_w_idxs
        doc_len, query_len = doc_mask.sum(-1), query_mask.sum(-1)
        
        # encoder
        u_doc, u_query = self.encoder(doc_w_idxs, query_w_idxs, doc_len, query_len)
        
        # co_attention
        p_prime, q_prime = self.co_attention(u_doc, u_query, doc_mask, query_mask)
    
        # self_attention
        d_double_prime, q_bold, p_bold = self.self_attention(p_prime, q_prime, doc_len, query_len, doc_mask, query_mask)

        # match
        p_start, p_end = self.match(d_double_prime, q_bold, doc_mask)
        
        #=============================================================
        # answer module (test)
#         p_start, p_end = self.answer_module(d_double_prime, q_bold, doc_mask)
        
        #=============================================================
        
        # cnn classifier
        # uncomment line below to use cnn classifier
#         print("cnn")
        # pred_score = self.cnn_classifier(p_prime, q_prime)
        
        # classifier
        pred_score = self.classifier(p_bold, q_bold)     
        
        p_start, p_end = self.match(d_double_prime, q_bold, doc_mask)
        
        return p_start, p_end, pred_score

       


    
    
    
    