import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

from layers.Embed import (
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_pos_temp,
    DataEmbedding_wo_temp,
)
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    my_Layernorm,
    series_decomp,
    series_decomp_multi,
)

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FEDFormer(nn.Module):
    """
    FEDFormer performs the attention mechanism in the frequency domain, achieving O(N) complexity.

    Attributes:
        version (str): The version of the model ('Fourier' or 'Wavelets').
        mode_select (str): Method to select modes in attention mechanisms.
        modes (int): Number of modes to use in frequency domain.
        seq_len (int): Length of the input sequence.
        label_len (int): Length of the label sequence.
        pred_len (int): Length of the prediction sequence.
        output_attention (bool): Flag to output attention weights.
        decomp (nn.Module): Decomposition module for input series.
        enc_embedding (nn.Module): Encoder embedding layer.
        dec_embedding (nn.Module): Decoder embedding layer.
        encoder (Encoder): Encoder component of the model.
        decoder (Decoder): Decoder component of the model.
    """

    def __init__(self, configs: object):
        """
        Initializes the FEDFormer model.

        Args:
            configs (object): Configuration object containing model hyperparameters.
        """
        super(FEDFormer, self).__init__()
        self.version: str = configs.version
        self.mode_select: str = configs.mode_select
        self.modes: int = configs.modes
        self.seq_len: int = configs.seq_len
        self.label_len: int = configs.label_len
        self.pred_len: int = configs.pred_len
        self.output_attention: bool = configs.output_attention

        # Device configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Decomposition
        self.decomp = self._init_decomposition(configs.moving_avg)

        # Embedding
        self.enc_embedding, self.dec_embedding = self._init_embeddings(configs)

        # Attention Mechanism
        encoder_self_att, decoder_self_att, decoder_cross_att = (
            self._init_attention_mechanism(configs)
        )

        # Encoder
        self.encoder = self._build_encoder(configs, encoder_self_att)

        # Decoder
        self.decoder = self._build_decoder(configs, decoder_self_att, decoder_cross_att)

    def _init_decomposition(self, moving_avg: Union[int, List[int]]) -> nn.Module:
        """
        Initializes the decomposition module based on the moving average kernel size.

        Args:
            moving_avg (int or list of int): Kernel size for moving average.

        Returns:
            nn.Module: Decomposition module.
        """
        if isinstance(moving_avg, list):
            decomp = series_decomp_multi(moving_avg)
            logger.info(
                "Initialized series_decomp_multi with kernel sizes: {}".format(
                    moving_avg
                )
            )
        else:
            decomp = series_decomp(moving_avg)
            logger.info(
                "Initialized series_decomp with kernel size: {}".format(moving_avg)
            )
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
            2: DataEmbedding_wo_pos_temp,
            3: DataEmbedding_wo_temp,
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

    def _init_attention_mechanism(
        self, configs: object
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """
        Initializes the attention mechanisms for encoder and decoder based on the model version.

        Args:
            configs (object): Configuration object.

        Returns:
            Tuple[nn.Module, nn.Module, nn.Module]: Encoder self-attention, decoder self-attention, decoder cross-attention modules.
        """
        if self.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base
            )
            decoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base
            )
            decoder_cross_att = MultiWaveletCross(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=configs.modes,
                ich=configs.d_model,
                base=configs.base,
                activation=configs.cross_activation,
            )
            logger.info("Initialized Wavelets-based attention mechanisms.")
        elif self.version == "Fourier":
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
            decoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.seq_len // 2 + self.pred_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
            logger.info("Initialized Fourier-based attention mechanisms.")
        else:
            raise ValueError(f"Unsupported version: {self.version}")

        return encoder_self_att, decoder_self_att, decoder_cross_att

    def _build_encoder(self, configs: object, encoder_self_att: nn.Module) -> Encoder:
        """
        Builds the encoder component.

        Args:
            configs (object): Configuration object.
            encoder_self_att (nn.Module): Encoder self-attention module.

        Returns:
            Encoder: Configured encoder.
        """
        encoder_layers = [
            EncoderLayer(
                AutoCorrelationLayer(
                    encoder_self_att, configs.d_model, configs.n_heads
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

    def _build_decoder(
        self,
        configs: object,
        decoder_self_att: nn.Module,
        decoder_cross_att: nn.Module,
    ) -> Decoder:
        """
        Builds the decoder component.

        Args:
            configs (object): Configuration object.
            decoder_self_att (nn.Module): Decoder self-attention module.
            decoder_cross_att (nn.Module): Decoder cross-attention module.

        Returns:
            Decoder: Configured decoder.
        """
        decoder_layers = [
            DecoderLayer(
                self_attention=AutoCorrelationLayer(
                    decoder_self_att, configs.d_model, configs.n_heads
                ),
                cross_attention=AutoCorrelationLayer(
                    decoder_cross_att, configs.d_model, configs.n_heads
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
        Forward pass of the FEDFormer model.

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
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len :, :], (0, 0, 0, self.pred_len)
        )

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, enc_attns = self.encoder(enc_out, attn_mask=enc_self_mask)

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
            return dec_out, enc_attns
        return dec_out  # Shape: [Batch, Pred_Len, Features]
