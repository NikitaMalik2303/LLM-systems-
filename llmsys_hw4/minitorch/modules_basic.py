"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weights : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN ASSIGN3_2
        self.weights= Parameter(tensor_from_numpy(np.random.rand(num_embeddings, embedding_dim), backend=self.backend, requires_grad=True))
        ### END ASSIGN3_2
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN ASSIGN3_2
        # X (b, seq_len) -> X(B, len, vocab)
        x= one_hot(x, self.num_embeddings)
        x= x.view(bs*seq_len, self.num_embeddings)
        out= x@self.weights.value
        out= out.view(bs, seq_len, self.embedding_dim)
        return out
        ### END ASSIGN3_2

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)

        Note: If p_dropout is 0, directly return the input tensor. Otherwise, the random seed may cause problems
        """
        ### BEGIN ASSIGN3_2
        if self.p_dropout==0 or (not self.training): 
            return x

        mask= tensor(np.random.binomial(1, 1-self.p_dropout, size=x.shape).tolist(), backend=x.backend)
        scale= tensor([1.0/ (1.0- float(self.p_dropout))], backend=x.backend)
        return x*mask*scale

        ### END ASSIGN3_2


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        self.out_size = out_size
        ### BEGIN ASSIGN3_2
        self.use_bias= bias
        
        limit= 1.0/np.sqrt(in_size)
        self.weights= Parameter(tensor_from_numpy(np.random.uniform(-limit, limit, (in_size, out_size)), backend= backend, requires_grad=True))
        self.bias= Parameter(tensor_from_numpy(np.random.uniform(-limit, limit, (out_size,)), backend=backend, requires_grad=True))
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        ### BEGIN ASSIGN3_2
        out= x@self.weights.value
        if self.use_bias: 
            out+= self.bias.value
        return out


        ### END ASSIGN3_2



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN ASSIGN3_2
        self.weights= Parameter(ones((dim,), backend=backend))
        self.bias= Parameter(zeros((dim,), backend=backend))
        ### END ASSIGN3_2

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        ### BEGIN ASSIGN3_2
        mean= x.mean(dim=1).view(batch, 1)
        centered= x- mean
        var= (centered*centered).mean(dim=1).view(batch, 1)
        std= (var+ tensor([float(self.eps)], backend=x.backend))**0.5

        x_norm= centered/ std
        return x_norm*self.weights.value + self.bias.value

        ### END ASSIGN3_2
