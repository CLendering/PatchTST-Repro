import torch
import torch.nn as nn
from typing import Optional, Tuple, List

import numpy as np
from layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    ConvLayer,
)
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import (
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_temp,
    DataEmbedding_wo_pos_temp,
)


class Model(nn.Module):
    """
    Informer model with ProbSparse attention for efficient sequence forecasting.

    Attributes:
        pred_len (int): The prediction length.
        output_attention (bool): Flag to output attention weights.
        enc_embedding (nn.Module): Encoder embedding layer.
        dec_embedding (nn.Module): Decoder embedding layer.
        encoder (Encoder): Encoder component of the model.
        decoder (Decoder): Decoder component of the model.
    """

    def __init__(self, configs: object):
        """
        Initializes the Model.

        Args:
            configs (object): Configuration object containing model hyperparameters.
        """
        super(Model, self).__init__()
        self.pred_len: int = configs.pred_len
        self.output_attention: bool = configs.output_attention

        # Initialize embeddings
        self.enc_embedding, self.dec_embedding = self._init_embeddings(configs)

        # Initialize encoder
        self.encoder = self._build_encoder(configs)

        # Initialize decoder
        self.decoder = self._build_decoder(configs)

    def _init_embeddings(self, configs: object) -> Tuple[nn.Module, nn.Module]:
        """
        Initializes encoder and decoder embeddings based on embed_type.

        Args:
            configs (object): Configuration object.

        Returns:
            Tuple[nn.Module, nn.Module]: Encoder and decoder embedding layers.
        """
        embedding_classes = {
            0: DataEmbedding,
            1: DataEmbedding,
            2: DataEmbedding_wo_pos,
            3: DataEmbedding_wo_temp,
            4: DataEmbedding_wo_pos_temp,
        }

        embed_cls = embedding_classes.get(configs.embed_type)
        if embed_cls is None:
            raise ValueError(f"Unsupported embed_type: {configs.embed_type}")

        enc_embedding = embed_cls(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        dec_embedding = embed_cls(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        return enc_embedding, dec_embedding

    def _build_encoder(self, configs: object) -> Encoder:
        """
        Builds the encoder component.

        Args:
            configs (object): Configuration object.

        Returns:
            Encoder: Configured encoder.
        """
        encoder_layers = [
            EncoderLayer(
                AttentionLayer(
                    ProbAttention(
                        mask_flag=False,
                        factor=configs.factor,
                        attention_dropout=configs.dropout,
                        output_attention=configs.output_attention,
                    ),
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                ),
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.e_layers)
        ]

        conv_layers = (
            [ConvLayer(configs.d_model) for _ in range(configs.e_layers - 1)]
            if configs.distil
            else None
        )

        encoder = Encoder(
            layers=encoder_layers,
            conv_layers=conv_layers,
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        return encoder

    def _build_decoder(self, configs: object) -> Decoder:
        """
        Builds the decoder component.

        Args:
            configs (object): Configuration object.

        Returns:
            Decoder: Configured decoder.
        """
        decoder_layers = [
            DecoderLayer(
                self_attention=AttentionLayer(
                    ProbAttention(
                        mask_flag=True,
                        factor=configs.factor,
                        attention_dropout=configs.dropout,
                        output_attention=False,
                    ),
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                ),
                cross_attention=AttentionLayer(
                    ProbAttention(
                        mask_flag=False,
                        factor=configs.factor,
                        attention_dropout=configs.dropout,
                        output_attention=False,
                    ),
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                ),
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.d_layers)
        ]

        decoder = Decoder(
            layers=decoder_layers,
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

        return decoder

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None,
        dec_enc_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass of the model.

        Args:
            x_enc (torch.Tensor): Encoder input tensor.
            x_mark_enc (torch.Tensor): Encoder time features.
            x_dec (torch.Tensor): Decoder input tensor.
            x_mark_dec (torch.Tensor): Decoder time features.
            enc_self_mask (Optional[torch.Tensor], optional): Encoder self-attention mask. Defaults to None.
            dec_self_mask (Optional[torch.Tensor], optional): Decoder self-attention mask. Defaults to None.
            dec_enc_mask (Optional[torch.Tensor], optional): Decoder-encoder attention mask. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
                - Predicted output tensor.
                - Attention weights (if output_attention is True).
        """
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
        )

        # Select the last pred_len steps
        dec_out = dec_out[:, -self.pred_len :, :]

        if self.output_attention:
            return dec_out, attns
        return dec_out  # Shape: [Batch, Pred_Len, Dim]
