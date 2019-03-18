import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearSeqAtt(nn.Module):
    def __init__(self, input_dim1, input_dim2):
        super(BilinearSeqAtt, self).__init__()

        self.linear = nn.Linear(input_dim1, input_dim2, bias = False)

    def forward(self, x, y):
#         print("x:",x.shape)
        xW = self.linear(x)
#         print("xW: ", xW.shape) # [64, 100, 640]
        xWy = torch.bmm(xW, y.permute(0,2,1))
#         print("xWy: ", xWy.shape) # [64, 100, 100]
        return xWy

class BilinearMatrixAttention(nn.Module):
    """
    Computes attention between two matrices using a bilinear attention function.  This function has
    a matrix of weights ``W``  and the similarity between the two matrices ``X`` and ``Y`` is computed as ``X W Y^T``.
    -------------
    label_dim : ``int``, optional (default = 1)
        The number of output classes. Typically in an attention setting this will be one,
        but this parameter allows this class to function as an equivalent to ``torch.nn.Bilinear``
        for matrices, rather than vectors.
    """
    def __init__(self,
                 matrix_1_dim: int,
                 matrix_2_dim: int,
                 label_dim: int = 1) -> None:
        super().__init__()

        if label_dim == 1:
            self._weight_matrix = nn.Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))
        else:
            self._weight_matrix = nn.Parameter(torch.Tensor(label_dim, matrix_1_dim, matrix_2_dim))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:

        weight = self._weight_matrix
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
        final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2, 3))
        
        return final.squeeze(1) 