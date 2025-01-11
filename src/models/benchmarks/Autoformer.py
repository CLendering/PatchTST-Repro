import logging
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import (
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_pos_temp,
    DataEmbedding_wo_temp,
)
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    my_Layernorm,
    series_decomp,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoFormer(nn.Module):
    """
    AutoFormer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity.

    Attributes:
        seq_len (int): Length of the input sequence.
        label_len (int): Length of the label sequence.
        pred_len (int): Length of the prediction sequence.
        output_attention (bool): Flag to output attention weights.
        decomp (nn.Module): Decomposition module for input series.
        enc_embedding (nn.Module): Encoder embedding layer.
        dec_embedding (nn.Module): Decoder embedding layer.
        encoder (Encoder): Encoder component of the model.
        decoder (Decoder): Decoder component of the model.
        device (torch.device): Device on which the model is running.
    """

    def __init__(self, configs: object):
        """
        Initializes the AutoFormer model.

        Args:
            configs (object): Configuration object containing model hyperparameters.
        """
        super(AutoFormer, self).__init__()
        self.seq_len: int = configs.seq_len
        self.label_len: int = configs.label_len
        self.pred_len: int = configs.pred_len
        self.output_attention: bool = configs.output_attention

        # Device configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Decomposition
        self.decomp = self._init_decomposition(configs.moving_avg)

        # Embedding
        self.enc_embedding, self.dec_embedding = self._init_embeddings(configs)

        # Encoder
        self.encoder = self._build_encoder(configs)

        # Decoder
        self.decoder = self._build_decoder(configs)

    def _init_decomposition(self, moving_avg: Union[int, List[int]]) -> nn.Module:
        """
        Initializes the decomposition module based on the moving average kernel size.

        Args:
            moving_avg (int or list of int): Kernel size for moving average.

        Returns:
            nn.Module: Decomposition module.
        """
        if isinstance(moving_avg, list):
            decomp = series_decomp(moving_avg)
            logger.info(f"Initialized series_decomp with kernel sizes: {moving_avg}")
        else:
            decomp = series_decomp(moving_avg)
            logger.info(f"Initialized series_decomp with kernel size: {moving_avg}")
        return decomp

    def _init_embeddings(self, configs: object) -> Tuple[nn.Module, nn.Module]:
        """
        Initializes encoder and decoder embeddings based on embed_type.

        Args:
            configs (object): Configuration object.

        Returns:
            Tuple[nn.Module, nn.Module]: Encoder and decoder embedding layers.
        """
        embedding_classes = {
            0: DataEmbedding_wo_pos,
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

        logger.info(f"Initialized embeddings with embed_type: {configs.embed_type}")
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
                AutoCorrelationLayer(
                    AutoCorrelation(
                        mask_flag=False,
                        factor=configs.factor,
                        attention_dropout=configs.dropout,
                        output_attention=configs.output_attention,
                    ),
                    configs.d_model,
                    configs.n_heads,
                ),
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                moving_avg=configs.moving_avg,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.e_layers)
        ]

        encoder = Encoder(
            layers=encoder_layers,
            norm_layer=my_Layernorm(configs.d_model),
        )

        logger.info(f"Built encoder with {configs.e_layers} layers.")
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
                self_attention=AutoCorrelationLayer(
                    AutoCorrelation(
                        mask_flag=True,
                        factor=configs.factor,
                        attention_dropout=configs.dropout,
                        output_attention=False,
                    ),
                    configs.d_model,
                    configs.n_heads,
                ),
                cross_attention=AutoCorrelationLayer(
                    AutoCorrelation(
                        mask_flag=False,
                        factor=configs.factor,
                        attention_dropout=configs.dropout,
                        output_attention=False,
                    ),
                    configs.d_model,
                    configs.n_heads,
                ),
                d_model=configs.d_model,
                c_out=configs.c_out,
                d_ff=configs.d_ff,
                moving_avg=configs.moving_avg,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.d_layers)
        ]

        decoder = Decoder(
            layers=decoder_layers,
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

        logger.info(f"Built decoder with {configs.d_layers} layers.")
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
        Forward pass of the AutoFormer model.

        Args:
            x_enc (torch.Tensor): Encoder input tensor of shape [Batch, Seq_Len, Features].
            x_mark_enc (torch.Tensor): Encoder time features.
            x_dec (torch.Tensor): Decoder input tensor of shape [Batch, Seq_Len, Features].
            x_mark_dec (torch.Tensor): Decoder time features.
            enc_self_mask (Optional[torch.Tensor], optional): Encoder self-attention mask. Defaults to None.
            dec_self_mask (Optional[torch.Tensor], optional): Decoder self-attention mask. Defaults to None.
            dec_enc_mask (Optional[torch.Tensor], optional): Decoder-encoder attention mask. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
                - Predicted output tensor of shape [Batch, Pred_Len, Features].
                - Attention weights (if output_attention is True).
        """
        # Move tensors to the appropriate device
        x_enc = x_enc.to(self.device)
        x_mark_enc = x_mark_enc.to(self.device)
        x_dec = x_dec.to(self.device)
        x_mark_dec = x_mark_dec.to(self.device)

        # Decomposition
        seasonal_init, trend_init = self.decomp(x_enc)

        # Prepare trend for decoder input
        mean = torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)

        # Prepare seasonal for decoder input
        zeros = torch.zeros(
            (x_dec.size(0), self.pred_len, x_dec.size(2)),
            device=self.device,
        )
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len :, :], zeros], dim=1
        )

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            trend=trend_init,
        )

        # Final output
        dec_out = trend_part + seasonal_part
        dec_out = dec_out[:, -self.pred_len :, :]

        if self.output_attention:
            return dec_out, attns
        return dec_out  # Shape: [Batch, Pred_Len, Features]
