__all__ = ["PatchTST_backbone"]

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor

from src.models.patchTST.positional.encoding import generate_positional_encoding
from src.models.patchTST.heads.flatten import FlattenHead
from src.models.patchTST.encoders.transformer import TSTEncoder
from src.models.patchTST.revin.revin import RevIN


class PatchTST_backbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        max_seq_len: Optional[int] = 1024,
        n_layers: int = 3,
        d_model=128,
        n_heads=16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout=0,
        padding_patch=None,
        pretrain_head: bool = False,
        head_type="flatten",
        individual=False,
        revin=True,
        affine=True,
        subtract_last=False,
        verbose: bool = False,
        only_patching=False,
        **kwargs,
    ):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(
            c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
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
            verbose=verbose,
            only_patching=only_patching,
            **kwargs,
        )

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(
                self.head_nf, c_in, fc_dropout
            )  # custom head passed as a partial func with all its kwargs
        elif head_type == "flatten":
            self.head = FlattenHead(
                self.individual,
                self.n_vars,
                self.head_nf,
                target_window,
                head_dropout=head_dropout,
            )

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

        # model
        z = self.backbone(z)  # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)  # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(
        self,
        c_in,
        patch_num,
        patch_len,
        max_seq_len=1024,
        n_layers=3,
        d_model=128,
        n_heads=16,
        d_k=None,
        d_v=None,
        d_ff=256,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        act="gelu",
        store_attn=False,
        key_padding_mask="auto",
        padding_var=None,
        attn_mask=None,
        res_attention=True,
        pre_norm=False,
        pe="zeros",
        learn_pe=True,
        verbose=False,
        only_patching=False,
        **kwargs,
    ):

        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len
        self.only_patching = only_patching

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(
            patch_len, d_model
        )  # Eq 1: projection of feature vectors onto a d-dim vector space

        # If only_patching=True, the "time length" is nvars * patch_num,
        # otherwise it's just patch_num:
        if only_patching:
            q_len = c_in * patch_num
        else:
            q_len = patch_num

        # Positional encoding
        self.W_pos = generate_positional_encoding(
            encoding_type=pe,
            sequence_length=q_len,
            model_dim=d_model,
            learnable=learn_pe,
        )

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            model_dim=d_model,
            num_heads=n_heads,
            feedforward_dim=d_ff,
            use_batch_norm=norm == "BatchNorm",
            attention_dropout=attn_dropout,
            dropout_rate=dropout,
            activation=act,
            residual_attention=res_attention,
            num_layers=n_layers,
            pre_normalization=pre_norm,
            store_attention=store_attn,
        )

    def forward(self, x: Tensor) -> Tensor:
        bs, nvars, patch_len, patch_num = x.shape

        # 1) Project: [bs, nvars, patch_num, d_model]
        x = x.permute(0, 1, 3, 2)
        x = self.W_P(x)

        # 2) Flatten
        if not self.only_patching:
            x = x.reshape(bs * nvars, patch_num, x.shape[-1])   # shape [bs*nvars, patch_num, d_model]
            pe_slice = self.W_pos[:patch_num, :]                # [patch_num, d_model]
        else:
            x = x.reshape(bs, patch_num * nvars, x.shape[-1])   # shape [bs, patch_num*nvars, d_model]
            pe_slice = self.W_pos[:(patch_num * nvars), :]       # [patch_num*nvars, d_model]

        # 3) Add positional encoding
        #    We'll unsqueeze(0) so pe_slice can broadcast across the batch dimension
        x = x + pe_slice.unsqueeze(0)  # shape => broadcast to [batch, seq_len, d_model]
        x = self.dropout(x)

        # 4) Transformer pass => shape still [batch, seq_len, d_model]
        x = self.encoder(x)

        # 5) Un-flatten
        if not self.only_patching:
            x = x.reshape(bs, nvars, patch_num, x.shape[-1])  # [bs, nvars, patch_num, d_model]
        else:
            x = x.reshape(bs, patch_num, nvars, x.shape[-1])  # [bs, patch_num, nvars, d_model]
            x = x.permute(0, 2, 1, 3)                         # => [bs, nvars, patch_num, d_model]

        return x
