import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from src.models.patchTST.attention.multihead import MultiheadAttention
from src.models.utils.tensor_ops import Transpose


class TSTEncoderLayer(nn.Module):
    """
    Single encoder layer comprising multi-head self-attention and a position-wise feed-forward network.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int = 256,
        use_batch_norm: bool = True,
        attention_dropout: float = 0.0,
        dropout_rate: float = 0.0,
        bias: bool = True,
        activation: str = "gelu",
        residual_attention: bool = False,
        pre_normalization: bool = False,
        store_attention: bool = False,
    ):
        super().__init__()

        # Ensure model_dim is divisible by num_heads
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})."
            )

        d_k = model_dim // num_heads
        d_v = model_dim // num_heads

        # Multi-head self-attention
        self.attention_block = MultiheadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim_q=d_k,  # Corrected argument name
            head_dim_v=d_v,  # Corrected argument name
            attention_dropout=attention_dropout,  # Corrected argument name
            projection_dropout=dropout_rate,  # Corrected argument name
            residual_attention=residual_attention,
        )

        # Residual and normalization for attention
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.pre_normalization = pre_normalization
        self.store_attention = store_attention
        self.residual_attention = residual_attention

        if use_batch_norm:
            self.attn_normalization = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(model_dim), Transpose(1, 2)
            )
        else:
            self.attn_normalization = nn.LayerNorm(model_dim)

        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim, bias=bias),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dim, model_dim, bias=bias),
        )
        self.ffn_dropout = nn.Dropout(dropout_rate)

        # Residual and normalization for feed-forward
        if use_batch_norm:
            self.ffn_normalization = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(model_dim), Transpose(1, 2)
            )
        else:
            self.ffn_normalization = nn.LayerNorm(model_dim)

    def forward(self, inputs: Tensor, previous_attention: Optional[Tensor] = None):
        """
        Forward pass for a single encoder layer.

        Args:
            inputs (Tensor): Shape [batch_size, sequence_length, model_dim].
            previous_attention (Optional[Tensor]): Previous attention scores if residual attention is used.

        Returns:
            Tuple[Tensor, Optional[Tensor]]:
                - Output of shape [batch_size, sequence_length, model_dim].
                - Updated attention scores if residual attention is enabled.
        """
        # Multi-head self-attention
        if self.pre_normalization:
            inputs = self.attn_normalization(inputs)

        if self.residual_attention:
            attended, attn_weights, scores = self.attention_block(
                inputs, inputs, inputs, previous_attention
            )
        else:
            attended, attn_weights = self.attention_block(inputs, inputs, inputs)
        if self.store_attention:
            # You can store attention weights or scores for analysis if needed
            self.last_attention = attn_weights

        # Residual connection + dropout
        inputs = inputs + self.attn_dropout(attended)
        if not self.pre_normalization:
            inputs = self.attn_normalization(inputs)

        # Feed-forward
        if self.pre_normalization:
            inputs = self.ffn_normalization(inputs)

        ff_out = self.feed_forward(inputs)

        # Residual connection + dropout
        outputs = inputs + self.ffn_dropout(ff_out)
        if not self.pre_normalization:
            outputs = self.ffn_normalization(outputs)

        if self.residual_attention:
            return outputs, scores
        else:
            return outputs, None


class TSTEncoder(nn.Module):
    """
    Stacks multiple TimeSeriesTransformerEncoderLayer modules.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int = 256,
        use_batch_norm: bool = True,
        attention_dropout: float = 0.0,
        dropout_rate: float = 0.0,
        activation: str = "gelu",
        residual_attention: bool = False,
        num_layers: int = 1,
        pre_normalization: bool = False,
        store_attention: bool = False,
    ):
        super().__init__()

        self.residual_attention = residual_attention
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    use_batch_norm=use_batch_norm,
                    attention_dropout=attention_dropout,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    residual_attention=residual_attention,
                    pre_normalization=pre_normalization,
                    store_attention=store_attention,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through multiple encoder layers.

        Args:
            inputs (Tensor): Shape [batch_size, seq_length, model_dim].

        Returns:
            Tensor: Shape [batch_size, seq_length, model_dim].
        """
        output = inputs
        attention_scores = None

        # If residual attention is enabled, pass scores through layers
        if self.residual_attention:
            for layer in self.layers:
                output, attention_scores = layer(
                    output, previous_attention=attention_scores
                )
            return output
        else:
            for layer in self.layers:
                output, _ = layer(output)
            return output
