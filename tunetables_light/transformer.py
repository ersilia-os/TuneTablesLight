import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerEncoder

from tunetables_light.layer import TransformerEncoderLayer
from tunetables_light.utils import SeqBN, bool_mask_to_att_mask


class TransformerModel(nn.Module):
    def __init__(
        self,
        encoder,
        n_out,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0.0,
        style_encoder=None,
        y_encoder=None,
        pos_encoder=None,
        decoder=None,
        input_normalization=False,
        init_method=None,
        pre_norm=False,
        activation="gelu",
        recompute_attn=False,
        num_global_att_tokens=0,
        full_attention=False,
        all_layers_same_init=False,
        efficient_eval_masking=True,
        prefix_size=0,
        n_classes=2,
        prefix_label_probs=None,
        num_features=100,
        linear=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.model_type = "Transformer"
        encoder_layer_creator = lambda: TransformerEncoderLayer(
            ninp,
            nhead,
            nhid,
            self.dropout,
            activation=activation,
            pre_norm=pre_norm,
            recompute_attn=recompute_attn,
            linear=linear,
        )
        self.transformer_encoder = (
            TransformerEncoder(encoder_layer_creator(), nlayers)
            if all_layers_same_init
            else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        )
        self.ninp = ninp
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder
        self.linear = linear
        self.decoder = (
            decoder(ninp, nhid, n_out)
            if decoder is not None
            else nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        )
        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.style_encoder = style_encoder
        self.init_method = init_method
        if num_global_att_tokens is not None:
            assert not full_attention
        self.global_att_embeddings = (
            nn.Embedding(num_global_att_tokens, ninp) if num_global_att_tokens else None
        )
        self.prefix_size = prefix_size
        if self.prefix_size > 0:
            self.prefix_embedding = nn.Embedding(prefix_size, ninp)
            if prefix_label_probs is not None:
                self.prefix_y_embedding = torch.multinomial(
                    prefix_label_probs, prefix_size, replacement=True
                )
            else:
                self.prefix_y_embedding = torch.randint(0, n_classes, (prefix_size,))
        self.full_attention = full_attention
        self.efficient_eval_masking = efficient_eval_masking

        self.n_out = n_out
        self.nhid = nhid

        self.init_weights()
    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("efficient_eval_masking", False)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        train_size = sz - query_size
        mask = torch.zeros(sz, sz) == 0
        mask[:, train_size:].zero_()
        mask |= torch.eye(sz) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_query_matrix(
        num_global_att_tokens, seq_len, num_query_tokens
    ):
        train_size = seq_len + num_global_att_tokens - num_query_tokens
        sz = seq_len + num_global_att_tokens
        mask = torch.zeros(num_query_tokens, sz) == 0
        mask[:, train_size:].zero_()
        mask[:, train_size:] |= torch.eye(num_query_tokens) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_trainset_matrix(
        num_global_att_tokens, seq_len, num_query_tokens
    ):
        train_size = seq_len + num_global_att_tokens - num_query_tokens
        trainset_size = seq_len - num_query_tokens
        mask = torch.zeros(trainset_size, num_global_att_tokens) == 0
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_globaltokens_matrix(
        num_global_att_tokens, seq_len, num_query_tokens
    ):
        mask = (
            torch.zeros(
                num_global_att_tokens,
                num_global_att_tokens + seq_len - num_query_tokens,
            )
            == 0
        )
        return bool_mask_to_att_mask(mask)

    def freeze_parameters_except_prefix(self):
        for name, param in self.named_parameters():
            if "prefix_embedding" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def freeze_parameters_except_named(self, param_strings):
        for name, param in self.named_parameters():
            grad_reqd = False
            for s in param_strings:
                if s in name:
                    grad_reqd = True
                    break
            param.requires_grad = grad_reqd

    def init_weights(self):
        if self.init_method is not None:
            self.apply(self.init_method)
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            attns = (
                layer.self_attn
                if isinstance(layer.self_attn, nn.ModuleList)
                else [layer.self_attn]
            )
            if not self.linear:
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    nn.init.zeros_(attn.out_proj.bias)

    def init_prefix_weights(self):
        initrange = 1.0
        self.prefix_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None, single_eval_pos=None):
        assert isinstance(src, tuple), (
            "inputs (src) have to be given as (x,y) or (style,x,y) tuple"
        )

        if len(src) == 2:
            src = (None,) + src

        style_src, x_src, y_src = src
        x_src = self.encoder(x_src)

        if self.prefix_size > 0:
            single_eval_pos = single_eval_pos + self.prefix_size
            if len(x_src.shape) > len(self.prefix_embedding.weight.shape):
                x_src = torch.cat([self.prefix_embedding.weight.unsqueeze(1).repeat(1, x_src.shape[1], 1), x_src], 0)
            elif len(x_src.shape) == len(self.prefix_embedding.weight.shape):
                x_src = torch.cat([self.prefix_embedding.weight, x_src], 0)
            else:
                x_src = torch.cat([x_src.unsqueeze(1), self.prefix_embedding.weight], 0)
            if len(y_src.shape) > len(self.prefix_y_embedding.shape):
                y_src = torch.cat(
                    [
                        self.prefix_y_embedding.to(
                            self.prefix_embedding.weight.device
                        ).unsqueeze(1).repeat(1, y_src.shape[1]),
                        y_src,
                    ],
                    0,
                )
            elif len(y_src.shape) == len(self.prefix_y_embedding.shape):
                y_src = torch.cat(
                    [
                        self.prefix_y_embedding.to(self.prefix_embedding.weight.device),
                        y_src,
                    ],
                    0,
                )
            else:
                y_src = torch.cat(
                    [
                        y_src.unsqueeze(1),
                        self.prefix_y_embedding.to(self.prefix_embedding.weight.device),
                    ],
                    0,
                )

        y_src = self.y_encoder(
            y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src
        )

        style_src = (
            self.style_encoder(style_src).unsqueeze(0)
            if self.style_encoder
            else torch.tensor([], device=x_src.device)
        )
        global_src = (
            torch.tensor([], device=x_src.device)
            if self.global_att_embeddings is None
            else self.global_att_embeddings.weight.unsqueeze(1).repeat(
                1, x_src.shape[1], 1
            )
        )
        if src_mask is not None:
            assert self.global_att_embeddings is None or isinstance(src_mask, tuple)
        if src_mask is None:
            if self.global_att_embeddings is None:
                full_len = len(x_src) + len(style_src)
                if self.full_attention:
                    src_mask = bool_mask_to_att_mask(
                        torch.ones((full_len, full_len), dtype=torch.bool)
                    ).to(x_src.device)
                elif self.efficient_eval_masking:
                    src_mask = single_eval_pos + len(style_src)
                else:
                    src_mask = self.generate_D_q_matrix(
                        full_len, len(x_src) - single_eval_pos
                    ).to(x_src.device)
            else:
                src_mask_args = (
                    self.global_att_embeddings.num_embeddings,
                    len(x_src) + len(style_src),
                    len(x_src) + len(style_src) - single_eval_pos,
                )
                src_mask = (
                    self.generate_global_att_globaltokens_matrix(*src_mask_args).to(
                        x_src.device
                    ),
                    self.generate_global_att_trainset_matrix(*src_mask_args).to(
                        x_src.device
                    ),
                    self.generate_global_att_query_matrix(*src_mask_args).to(
                        x_src.device
                    ),
                )
        n = min(x_src.size(0), y_src.size(0), single_eval_pos)
        train_x = x_src[:n] + y_src[:n]
        src = torch.cat([global_src, style_src, train_x, x_src[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        if self.pos_encoder is not None:
            src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output[
            single_eval_pos
            + len(style_src)
            + (
                self.global_att_embeddings.num_embeddings
                if self.global_att_embeddings
                else 0
            ) :
        ]

    @torch.no_grad()
    def init_from_small_model(self, small_model):
        assert (
            isinstance(self.decoder, nn.Linear)
            and isinstance(self.encoder, (nn.Linear, nn.Sequential))
            and isinstance(self.y_encoder, (nn.Linear, nn.Sequential))
        )

        def set_encoder_weights(my_encoder, small_model_encoder):
            my_encoder_linear, small_encoder_linear = (
                (my_encoder, small_model_encoder)
                if isinstance(my_encoder, nn.Linear)
                else (my_encoder[-1], small_model_encoder[-1])
            )
            small_in_dim = small_encoder_linear.out_features
            my_encoder_linear.weight.zero_()
            my_encoder_linear.bias.zero_()
            my_encoder_linear.weight[:small_in_dim] = small_encoder_linear.weight
            my_encoder_linear.bias[:small_in_dim] = small_encoder_linear.bias

        set_encoder_weights(self.encoder, small_model.encoder)
        set_encoder_weights(self.y_encoder, small_model.y_encoder)

        small_in_dim = small_model.decoder.in_features

        self.decoder.weight[:, :small_in_dim] = small_model.decoder.weight
        self.decoder.bias = small_model.decoder.bias

        for my_layer, small_layer in zip(
            self.transformer_encoder.layers, small_model.transformer_encoder.layers
        ):
            small_hid_dim = small_layer.linear1.out_features
            my_in_dim = my_layer.linear1.in_features

            my_in_proj_w = my_layer.self_attn.in_proj_weight
            small_in_proj_w = small_layer.self_attn.in_proj_weight

            my_in_proj_w.view(3, my_in_dim, my_in_dim)[
                :, :small_in_dim, :small_in_dim
            ] = small_in_proj_w.view(3, small_in_dim, small_in_dim)
            my_layer.self_attn.in_proj_bias.view(3, my_in_dim)[:, :small_in_dim] = (
                small_layer.self_attn.in_proj_bias.view(3, small_in_dim)
            )

            my_layer.self_attn.out_proj.weight[:small_in_dim, :small_in_dim] = (
                small_layer.self_attn.out_proj.weight
            )
            my_layer.self_attn.out_proj.bias[:small_in_dim] = (
                small_layer.self_attn.out_proj.bias
            )

            my_layer.linear1.weight[:small_hid_dim, :small_in_dim] = (
                small_layer.linear1.weight
            )
            my_layer.linear1.bias[:small_hid_dim] = small_layer.linear1.bias

            my_layer.linear2.weight[:small_in_dim, :small_hid_dim] = (
                small_layer.linear2.weight
            )
            my_layer.linear2.bias[:small_in_dim] = small_layer.linear2.bias

            my_layer.norm1.weight[:small_in_dim] = (
                math.sqrt(small_in_dim / my_in_dim) * small_layer.norm1.weight
            )
            my_layer.norm2.weight[:small_in_dim] = (
                math.sqrt(small_in_dim / my_in_dim) * small_layer.norm2.weight
            )

            my_layer.norm1.bias[:small_in_dim] = small_layer.norm1.bias
            my_layer.norm2.bias[:small_in_dim] = small_layer.norm2.bias


class TransformerEncoderDiffInit(Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([
            encoder_layer_creator() for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
