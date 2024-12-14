r"""
An implementation of transformer encoders.
"""

from typing import Optional
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            H: int,
            D_e: int,
            D_qk: Optional[int] = None,
            D_v: Optional[int] = None,
        ):
        r""" Multi-head attention module with attention attribution.
        
        Args:
            H (int): The number of heads in the multi-head attention.
            D_e (int): The embedding dimension.
            D_qk (int): The dimension of the query/key embedding.
            D_v (int): The dimension of the value embedding.
        """

        super().__init__()

        self.H = H
        self.D_e = D_e
        self.D_qk = D_e if D_qk is None else D_qk
        self.D_v = D_e if D_v is None else D_v
        assert (D_qk % H == 0) and (D_v % H == 0), "D_qk and D_v must be divisible by H."
        self.Dh_qk = D_qk // H
        self.Dh_v = D_v // H
        self.W_Q = nn.Linear(D_e, D_qk, bias=False)
        self.W_K = nn.Linear(D_e, D_qk, bias=False)
        self.W_V = nn.Linear(D_e, D_v, bias=False)

        self.fc = nn.Linear(D_v, D_e, bias=False) # NOTE

    def attention(
            self,
            X_e: torch.Tensor,
            Xc_e: torch.Tensor,
    ):
        r""" 
        Returns the attention weights.
        Default on CPU.
        """
        N = X_e.shape[0]
        L = X_e.shape[1]
        L_c = Xc_e.shape[1]
        
        with torch.no_grad():
            # linear projection
            Q = self.W_Q(X_e) \
                .view(N, L, self.H, self.Dh_qk) \
                .transpose(1, 2) # (N, L, D_e) -> (N, L, D_qk) -> (N, L, H, Dh_qk) -> (N, H, L, Dh_qk)
            K = self.W_K(Xc_e) \
                .view(N, L_c, self.H, self.Dh_qk) \
                .transpose(1, 2) # (N, L_c, D_e) -> (N, L_c, D_qk) -> (N, L_c, H, Dh_qk) -> (N, H, L_c, Dh_qk)
            V = self.W_V(Xc_e) \
                .view(N, L_c, self.H, self.Dh_v) \
                .transpose(1, 2) # (N, L_c, D_e) -> (N, L_c, D_v) -> (N, L_c, H, Dh_v) -> (N, H, L_c, Dh_v)
            # scaled dot-product attention
            A = nn.Softmax(dim=-1)(torch.matmul(
                Q, K.transpose(-1, -2)) / np.sqrt(self.Dh_qk)) # (N, H, L, Dh_qk) * (N, H, Dh_qk, L_c) -> (N, H, L, L_c)
        return A

    def forward(
            self, 
            X_e: torch.Tensor,
            Xc_e: torch.Tensor,
        ):
        """
        # TODO double check QK row / col relationship

        Args:
            X_e (torch.Tensor): The query data embedding.
            Xc_e (torch.Tensor): The conditional data (keys and values) 
                embedding. If it is the same as X_e, then perform self-attention, 
                otherwise cross-attention.
        Shape:
            - X_e: :math:`(N, L, D_e)`. L is the sequence length.
            - Xc_e: :math:`(N, L_c, D_e)`. L_c is the sequence length of the conditional data.
        Returns:
            - X_e_output: :math:`(N, L, D_e)`. The output of the multi-head attention.
        """

        N = X_e.shape[0]
        L = X_e.shape[1]
        L_c = Xc_e.shape[1]
        
        # linear projection
        Q = self.W_Q(X_e) \
            .view(N, L, self.H, self.Dh_qk) \
            .transpose(1, 2) # (N, L, D_e) -> (N, L, D_qk) -> (N, L, H, Dh_qk) -> (N, H, L, Dh_qk)
        K = self.W_K(Xc_e) \
            .view(N, L_c, self.H, self.Dh_qk) \
            .transpose(1, 2) # (N, L_c, D_e) -> (N, L_c, D_qk) -> (N, L_c, H, Dh_qk) -> (N, H, L_c, Dh_qk)
        V = self.W_V(Xc_e) \
            .view(N, L_c, self.H, self.Dh_v) \
            .transpose(1, 2) # (N, L_c, D_e) -> (N, L_c, D_v) -> (N, L_c, H, Dh_v) -> (N, H, L_c, Dh_v)
        # scaled dot-product attention
        A = nn.Softmax(dim=-1)(torch.matmul(
            Q, K.transpose(-1, -2)) / np.sqrt(self.Dh_qk)) # (N, H, L, Dh_qk) * (N, H, Dh_qk, L_c) -> (N, H, L, L_c)
        X_e_context = torch.matmul(A, V) # (N, H, L, L_c) * (N, H, L_c, Dh_v) --> (N, H, L, Dh_v)
        # concatenate heads
        X_e_context = X_e_context.transpose(1, 2).reshape(N, L, self.D_v) # (N, H, L, Dh_v) --> (N, L, H, Dh_v) -> (N, L, D_v)
        # linear projection
        X_e_output = self.fc(X_e_context) # D_v to D_e: (N, L, D_v) -> (N, L, D_e)
        
        return X_e_output


class Residual(nn.Module):
    def __init__(
            self,
            D_e: int,
            dropout: float = 0.2
    ):
        super().__init__()
        self.norm = nn.LayerNorm(D_e) # RMSNorm(dim=D_e) LLaMA
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X_e: torch.Tensor,
            layer: nn.Module,
            **kwargs,
    ):
        r"""
        Args:
            X_e (torch.Tensor): The input data embedding.
            layer (nn.Module): The layer to be applied.
        Shape:
            - X_e: :math:`(N, L, D_e)`. L is the sequence length.
        Note:
            - Dropout should happen before adding the residual connection 
                (ref: https://github.com/feather-ai/transformers-tutorial/blob/main/layers/residual_layer_norm.py)
            - use pre norm (modern order): https://docs.google.com/presentation/d/1ZXFIhYczos679r70Yu8vV9uO6B1J0ztzeDxbnBxD1S0/edit#slide=id.g13dd67c5ab8_0_2543
                post norm (old order): self.norm(X_e + self.dropout(layer(X_e=X_e, **kwargs)))
                TODO double check if using pre norm, then where should dropout be placed?
        """
        return X_e + self.dropout(layer(X_e=self.norm(X_e), **kwargs))

class PositionwiseFeedforward(nn.Module):
    def __init__(
            self,
            D_e: int,
            dropout: float = 0.2
    ):
        r"""
        
        Args:
            D_e (int): The embedding dimension.
            dropout (float): The dropout rate.
        """
        super().__init__()
        D_ff = 4 * D_e # 512 * 4 = 2048 (transformer 2017) (4d); 8d/3 (LLaMA)
        self.fc1 = nn.Linear(D_e, D_ff)
        self.fc2 = nn.Linear(D_ff, D_e)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X_e: torch.Tensor,
    ):
        r"""
        Args:
            X_e (torch.Tensor): The input data embedding.
        Shape:
            - X_e: :math:`(N, L, D_e)`. L is the sequence length.
        """
        return self.fc2(self.dropout(F.relu(self.fc1(X_e))))

class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            H: int,
            D_e: int,
            D_qk: Optional[int] = None,
            D_v: Optional[int] = None,
            dropout: float = 0.2,
        ) -> torch.Tensor:
        super().__init__()

        self.D_e = D_e
        D_qk = D_e if D_qk is None else D_qk
        D_v = D_e if D_v is None else D_v

        # residual
        self.residual = Residual(
            D_e=D_e,
            dropout=dropout
        )
        # multi-head attention
        self.mha = MultiHeadAttention(
            H=H,
            D_e=D_e,
            D_qk=D_qk,
            D_v=D_v,
        )
        # feed forward
        self.ffn = PositionwiseFeedforward(
            D_e=D_e,
            dropout=dropout
        )

    def forward(
        self,
        X_e: torch.Tensor,
        Xc_e: torch.Tensor,
    ):
        r"""
        Args:
            X_e (torch.Tensor): The query data embedding.
            Xc_e (torch.Tensor): The conditional data (keys and values)
                embedding. If the same as X_e, then perform self-attention,
                otherwise cross-attention.
        Shape:
            - X_e: :math:`(N, L, D_e)`
            - Xc_e: :math:`(N, L_c, D_e)`
        """
        # mha
        X_e = self.residual(
            X_e = X_e,
            layer = self.mha,
            Xc_e = Xc_e
        )
        # feed forward
        X_e = self.residual(
            X_e = X_e,
            layer = self.ffn
        )
        return X_e

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            N: int,
            H: int,
            D_e: int,
            D_qk: Optional[int] = None,
            D_v: Optional[int] = None,
            dropout: float = 0.2,
        ):
        super().__init__()

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                H=H,
                D_e=D_e,
                D_qk=D_qk,
                D_v=D_v,
                dropout=dropout
            ) for _ in range(N)
        ])

    def forward(
        self,
        X_e: torch.Tensor,
        Xc_e: torch.Tensor,
    ):
        r"""
        Args:
            X_e (torch.Tensor): The query data embedding.
            Xc_e (torch.Tensor): The conditional data (keys and values)
                embedding. If the same as X_e, then perform self-attention,
                otherwise cross-attention.
        Shape:
            - X_e: :math:`(N, L, D_e)`
            - Xc_e: :math:`(N, L_c, D_e)`
        """
        for encoder_layer in self.encoder_layers:
            X_e = encoder_layer(X_e=X_e, Xc_e=Xc_e) # TODO NOTE: conditional embedding is used repeatedly.
        return X_e

class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization (RMSNorm).
    Adapted from LLaMA"""
    def __init__ (
        self,
        dim: int,
        eps: float=1e-6
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight # weights are gamma and bete, two learnable parameters
    
