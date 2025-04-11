__all__ = ["PatchTST"]

from typing import Optional
from torch import nn, Tensor

from src.models.patchTST.encoders.supervised_backbone import PatchTST_backbone
from src.models.benchmarks.layers.Autoformer_EncDec import series_decomp
from src.training.config import TrainingConfig


class PatchTST(nn.Module):
    """
    PatchTST model for time-series forecasting. Uses an optional decomposition
    module to split the input into trend and residual components, each processed
    by a dedicated PatchTST backbone.
    """

    def __init__(
        self,
        configs: TrainingConfig,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = "flatten",
        verbose: bool = False,
        **kwargs
    ):
        super().__init__()

        # Extract parameters from configs
        c_in = configs.encoder_input_size
        context_window = configs.input_length
        target_window = configs.prediction_length

        n_layers = configs.num_encoder_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_fcn
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual_head
        patch_len = configs.patch_length
        stride = configs.stride
        padding_patch = configs.patch_padding

        revin = configs.revin
        affine = configs.revin_affine
        subtract_last = configs.subtract_last

        self.decomposition = configs.decomposition
        self.kernel_size = configs.kernel_size

        # Build the model(s)
        if self.decomposition:
            # Trend/residual decomposition
            self.decomp_module = series_decomp(self.kernel_size)

            self.model_res = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                only_patching=configs.only_patching,
                **kwargs
            )

            self.model_trend = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                only_patching=configs.only_patching,
                **kwargs
            )
        else:
            # Single model without decomposition
            self.model = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                only_patching=configs.only_patching,
                **kwargs
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for PatchTST.

        Args:
            x: Tensor of shape [Batch, Input_length, Channels].

        Returns:
            Forecast of shape [Batch, Output_length, Channels].
        """
        if self.decomposition:
            # Decomposition into residual and trend
            res_init, trend_init = self.decomp_module(x)
            res_init = res_init.permute(0, 2, 1)  # -> [Batch, Channels, Input_length]
            trend_init = trend_init.permute(0, 2, 1)

            # Forward pass for residual and trend
            res_out = self.model_res(res_init)
            trend_out = self.model_trend(trend_init)

            # Combine and permute back
            x = (res_out + trend_out).permute(
                0, 2, 1
            )  # -> [Batch, Output_length, Channels]
        else:
            # No decomposition
            x = x.permute(0, 2, 1)
            x = self.model(x)
            x = x.permute(0, 2, 1)

        return x
