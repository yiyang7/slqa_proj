import torch
import torch.nn as nn
import torch.nn.functional as F
from util import masked_softmax
from allennlp.modules.input_variational_dropout import InputVariationalDropout

"""
---------------------------------------------------------------
Co_Attention

---------------------------------------------------------------
"""
class Co_Attention(nn.Module):
    def __init__(self, hidden_size, drop_prob=0.):
        super(Co_Attention, self).__init__()
        
        self.drop_prob = drop_prob
        
        # For get_similarity_matrix_slqa
        self.shared_weight = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        nn.init.xavier_uniform_(self.shared_weight)
        
        # dropout mar 13 YL
        self._variational_dropout = InputVariationalDropout(drop_prob)
        
        # fuse
        self.linear_fuse_pq = nn.Linear(hidden_size*4,hidden_size,bias=True)
        self.linear_fuse_qp = nn.Linear(hidden_size*4,hidden_size,bias=True)
        self.tanh = nn.Tanh()
        
        # gate
        self.linear_gate_pq = nn.Linear(hidden_size*4,1,bias=True)
        self.linear_gate_qp = nn.Linear(hidden_size*4,1,bias=True)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, u_doc, u_query, doc_mask, query_mask):
        
        # Get P, Q, P_tilde, Q_tilde
        batch_size, doc_len, emb_size = u_doc.size()
        query_len = u_query.size(1)
        
        s = self.get_similarity_matrix_slqa(u_doc, u_query)
        
        # For Similar matrix
        doc_mask = doc_mask.view(batch_size, 1, doc_len)
        query_mask = query_mask.view(batch_size, query_len, 1)


        s_p2q = masked_softmax(s, query_mask, dim=1)
        s_q2p = masked_softmax(s, doc_mask, dim=2)

        q_tilde = torch.bmm(u_query.transpose(1,2), s_p2q).transpose(1,2)
        p_tilde = torch.bmm(s_q2p, u_doc)

        p = u_doc
        q = u_query

        # FUSE
        concat_input_fuse1 = torch.cat((p,q_tilde,p*q_tilde,p-q_tilde),2)
        concat_input_fuse2 = torch.cat((q,p_tilde,q*p_tilde,q-p_tilde),2)
        
        # dropout Mar 13 YL
        concat_input_fuse1 = self._variational_dropout(concat_input_fuse1)
        concat_input_fuse2 = self._variational_dropout(concat_input_fuse2)
        
        fuse_p_q_tilde = self.tanh(self.linear_fuse_pq(concat_input_fuse1))
        fuse_q_p_tilde = self.tanh(self.linear_fuse_qp(concat_input_fuse2))


        # GATE
        gate_p_q_tilde = self.sigmoid(self.linear_gate_pq(concat_input_fuse1))
        gate_q_p_tilde = self.sigmoid(self.linear_gate_qp(concat_input_fuse2))

        p_prime = gate_p_q_tilde*fuse_p_q_tilde + (1-gate_p_q_tilde)*p
        q_prime = gate_q_p_tilde*fuse_q_p_tilde + (1-gate_q_p_tilde)*q


        return p_prime, q_prime
        
        
    def get_similarity_matrix_slqa(self, u_doc, u_query):
        u_query_out = F.relu(torch.matmul(u_query, self.shared_weight))  # [64, 10, 1]
        u_doc_out = F.relu(torch.matmul(u_doc, self.shared_weight)).transpose(1, 2)  # [64, 1, 100]
        s = torch.bmm(u_query_out, u_doc_out)# [64, 10, 100]
        return s