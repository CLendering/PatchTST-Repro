import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor

import sys

sys.path.append(__file__.replace("src/models/encoders/patchTST.py", ""))

from src.models.attention import MultiheadAttention
from src.models.encoders.transformer import TSTEncoder
from src.models.positional import generate_positional_encoding
from src.models.heads import (
    LinearPretrainHead,
    LinearPredictionHead,
    LinearRegressionHead,
    LinearClassificationHead,
)


class PatchTST(nn.Module):
    """
    PatchTST Model for tasks such as prediction, regression, classification, and pretraining.

    Output Dimensions:
        - Prediction: [batch_size x target_dim x num_vars]
        - Regression: [batch_size x target_dim]
        - Classification: [batch_size x target_dim]
        - Pretraining: [batch_size x num_patches x num_vars x patch_length]
    """

    def __init__(
        self,
        input_channels: int,
        target_dim: int,
        patch_length: int,
        num_patches: int,
        num_layers: int = 3,
        model_dim: int = 128,
        num_heads: int = 8,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        head_type: str = "prediction",
        y_range: Optional[Tuple[float, float]] = None,
        **encoder_kwargs,
    ):
        """
        Initialize the PatchTST model.

        Args:
            input_channels (int): Number of input channels (variables).
            target_dim (int): Dimension of the target output.
            patch_length (int): Length of each patch.
            num_patches (int): Number of patches.
            num_layers (int, optional): Number of encoder layers. Defaults to 3.
            model_dim (int, optional): Dimension of the model. Defaults to 128.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            feedforward_dim (int, optional): Dimension of the feedforward network. Defaults to 256.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            activation (str, optional): Activation function. Defaults to "gelu".
            head_type (str, optional): Type of head ("pretrain", "prediction", "regression", "classification"). Defaults to "prediction".
            y_range (Optional[Tuple[float, float]], optional): Range for regression targets. Defaults to None.
            **encoder_kwargs: Additional keyword arguments for the encoder.
        """
        super(PatchTST, self).__init__()

        # Validate head type
        valid_head_types = {"pretrain", "prediction", "regression", "classification"}
        if head_type not in valid_head_types:
            raise ValueError(
                f"head_type must be one of {valid_head_types}, got '{head_type}'."
            )

        # Initialize the encoder backbone
        self.encoder = PatchTSTEncoder(
            input_channels=input_channels,
            num_patches=num_patches,
            patch_length=patch_length,
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            activation=activation,
            **encoder_kwargs,
        )

        # Initialize the appropriate head based on head_type
        self.head_type = head_type
        self.num_vars = input_channels
        self.target_dim = target_dim

        if head_type == "pretrain":
            self.head = LinearPretrainHead(
                model_dim=model_dim, patch_length=patch_length, dropout_rate=dropout
            )
        elif head_type == "prediction":
            self.head = LinearPredictionHead(
                num_vars=self.num_vars,
                model_dim=model_dim,
                num_patches=num_patches,
                forecast_len=target_dim,
                dropout_rate=dropout,
                individual=True,
            )
        elif head_type == "regression":
            self.head = LinearRegressionHead(
                num_vars=self.num_vars,
                model_dim=model_dim,
                output_dim=target_dim,
                dropout_rate=dropout,
                y_range=y_range,
            )
        elif head_type == "classification":
            self.head = LinearClassificationHead(
                num_vars=self.num_vars,
                model_dim=model_dim,
                num_classes=target_dim,
                dropout_rate=dropout,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchTST model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_channels, sequence_length].

        Returns:
            torch.Tensor: Output tensor based on the head type.
        """
        encoded_features = self.encoder(x)
        output = self.head(encoded_features)
        return output


class PatchTSTEncoder(nn.Module):
    """
    Encoder that processes time series data in patches, then applies a transformer-based architecture.
    """

    def __init__(
        self,
        input_channels: int,
        num_patches: int,
        patch_length: int,
        num_layers: int = 3,
        model_dim: int = 128,
        num_heads: int = 16,
        feedforward_dim: int = 256,
        use_batch_norm: bool = True,
        attention_dropout: float = 0.0,
        dropout_rate: float = 0.0,
        activation: str = "gelu",
        store_attention: bool = False,
        residual_attention: bool = True,
        pre_normalization: bool = False,
        positional_encoding_type: str = "zeros",
        learn_positional_encoding: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.num_channels = input_channels
        self.num_patches = num_patches
        self.patch_length = patch_length
        self.model_dim = model_dim
        self.shared_projection = kwargs.get("shared_projection", True)

        # Depending on whether embeddings are shared, create the projection layers
        if not self.shared_projection:
            # One projection per channel
            self.patch_projection = nn.ModuleList(
                [
                    nn.Linear(self.patch_length, self.model_dim)
                    for _ in range(self.num_channels)
                ]
            )
        else:
            # Single projection for all channels
            self.patch_projection = nn.Linear(self.patch_length, self.model_dim)

        # Positional encoding
        self.positional_encoding = generate_positional_encoding(
            encoding_type=positional_encoding_type,
            learnable=learn_positional_encoding,
            sequence_length=self.num_patches,
            model_dim=self.model_dim,
        )

        # Dropout for embeddings
        self.dropout = nn.Dropout(dropout_rate)

        # Transformer encoder (multiple layers)
        self.encoder = TSTEncoder(
            model_dim=self.model_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            use_batch_norm=use_batch_norm,
            attention_dropout=attention_dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            residual_attention=residual_attention,
            num_layers=num_layers,
            pre_normalization=pre_normalization,
            store_attention=store_attention,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Processes input patches through projection, applies positional encoding, then encodes with the transformer.

        Args:
            inputs (Tensor): Shape [batch_size, num_patches, num_channels, patch_length].

        Returns:
            Tensor: Encoded output of shape [batch_size, num_channels, model_dim, num_patches].
        """
        batch_size, num_patches, num_channels, patch_length = inputs.shape
        # Project each channel if embeddings are not shared
        if not self.shared_projection:
            projected_channels = []
            for channel_index in range(num_channels):
                embedded = self.patch_projection[channel_index](
                    inputs[:, :, channel_index, :]
                )
                projected_channels.append(embedded)
            # Recombine channels along dimension=2
            x = torch.stack(projected_channels, dim=2)
        else:
            # Use one projection layer for all channels
            x = self.patch_projection(inputs)

        # x shape: [batch_size, num_patches, num_channels, model_dim]
        x = x.transpose(1, 2)
        # x shape: [batch_size, num_channels, num_patches, model_dim]

        # Flatten batch and channel dimensions for transformer
        x_reshaped = x.reshape(batch_size * num_channels, num_patches, self.model_dim)

        # Add positional encoding
        x_reshaped = x_reshaped + self.positional_encoding
        x_reshaped = self.dropout(x_reshaped)

        # Pass through transformer
        encoded = self.encoder(x_reshaped)
        # encoded shape: [batch_size * num_channels, num_patches, model_dim]

        # Reshape back to [batch_size, num_channels, model_dim, num_patches]
        encoded = encoded.reshape(batch_size, num_channels, num_patches, self.model_dim)
        encoded = encoded.permute(0, 1, 3, 2)

        return encoded


def main():
    """
    Simple test script to verify PatchTST and its components.
    This script:
      1. Creates a synthetic batch of input data.
      2. Initializes a PatchTST model with different head types.
      3. Prints the output shapes to confirm correctness.
    """

    # Synthetic batch: [batch_size, num_patches, input_channels, patch_length]
    batch_size = 2
    num_patches = 3
    input_channels = 4
    patch_length = 5

    # Create random input
    # Shape must match PatchTSTEncoder: [bs, num_patches, num_channels, patch_len]
    x = torch.randn(batch_size, num_patches, input_channels, patch_length)

    # Test 1: Prediction head
    model_pred = PatchTST(
        input_channels=input_channels,
        target_dim=6,  # e.g. forecasting 6 future points
        patch_length=patch_length,
        num_patches=num_patches,
        head_type="prediction",  # set head
        model_dim=16,
        num_heads=4,
        feedforward_dim=32,
    )
    out_pred = model_pred(x)
    print("Prediction Output Shape:", out_pred.shape)

    # Test 2: Regression head
    model_reg = PatchTST(
        input_channels=input_channels,
        target_dim=1,  # e.g. univariate regression
        patch_length=patch_length,
        num_patches=num_patches,
        head_type="regression",
        model_dim=16,
        num_heads=4,
        feedforward_dim=32,
        y_range=(0.0, 1.0),  # map outputs to [0, 1]
    )
    out_reg = model_reg(x)
    print("Regression Output Shape:", out_reg.shape)

    # Test 3: Classification head
    model_cls = PatchTST(
        input_channels=input_channels,
        target_dim=10,  # e.g. 10 classes
        patch_length=patch_length,
        num_patches=num_patches,
        head_type="classification",
        model_dim=16,
        num_heads=4,
        feedforward_dim=32,
    )
    out_cls = model_cls(x)
    print("Classification Output Shape:", out_cls.shape)

    # Test 4: Pretraining head
    model_pre = PatchTST(
        input_channels=input_channels,
        target_dim=patch_length,  # not directly used by pretraining head
        patch_length=patch_length,
        num_patches=num_patches,
        head_type="pretrain",
        model_dim=16,
        num_heads=4,
        feedforward_dim=32,
    )
    out_pre = model_pre(x)
    print("Pretrain Output Shape:", out_pre.shape)


if __name__ == "__main__":
    main()
