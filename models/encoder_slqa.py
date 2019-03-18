import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RNNEncoder
from allennlp.modules.input_variational_dropout import InputVariationalDropout

"""
---------------------------------------------------------------
Encoder

---------------------------------------------------------------
"""
class Encoder(nn.Module):
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(Encoder, self).__init__()
        
        self.input_size_query = 300
        self.input_size_doc = 300
        self.embed_size = 300
        self.drop_prob = 0.5

        # Word embedding: glove
        self.glove_embedding = nn.Embedding.from_pretrained(word_vectors)
        
        
        # two Bi-LSTM
        self.bilstm1 = RNNEncoder(input_size=self.input_size_doc , hidden_size=hidden_size, drop_prob=drop_prob, num_layers = 1)   
        self.bilstm2 = RNNEncoder(input_size=self.input_size_query, hidden_size=hidden_size, drop_prob=drop_prob,num_layers = 1)   
        
        # dropout mar 13 YL
        self._variational_dropout = InputVariationalDropout(drop_prob)

    def forward(self, doc_w_idxs, query_w_idxs, doc_len, query_len):
        
        doc_word_emb = self.glove_embedding(doc_w_idxs)
        query_word_emb = self.glove_embedding(query_w_idxs)

        
        doc_word_emb = F.dropout(doc_word_emb, self.drop_prob, self.training)
        query_word_emb = F.dropout(query_word_emb, self.drop_prob, self.training)
        

        output_doc = self.bilstm1(doc_word_emb, doc_len)
        output_query = self.bilstm2(query_word_emb,query_len)  

        u_doc = torch.cat((output_doc, doc_word_emb),2)
        u_query = torch.cat((output_query, query_word_emb),2)
                        
        # dropout mar 13 YL
        u_doc = self._variational_dropout(u_doc)
        u_query = self._variational_dropout(u_query)
            
        return u_doc, u_query