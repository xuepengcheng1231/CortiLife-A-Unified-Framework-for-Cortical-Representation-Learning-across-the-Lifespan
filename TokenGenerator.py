import time

import torch
import numpy as np
from torch import nn
from einops import repeat, pack, unpack
from huggingface_hub import PyTorchModelHubMixin
from tokgen_utils.encoders import MultiScaledScalarEncoder, LinearEncoder
from vit_utils.positional_encoding import PositionalEncoding
from vit_utils.transformer import Transformer
from util.utils import get_neighs_order,Get_parcellation
from SphericalLayer.Layer import res_block, onering_conv_layer_batch
# ==================================
# ====       Organization:      ====
# ==================================
# ==== class TokenGeneratorUnit ====
# ==== class ViTUnit            ====
# ==== class Mantis8M           ====
# ==================================

class TokenGeneratorUnit(nn.Module):
    def __init__(self,  input_channel, hidden_dim, num_patches, scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc,
                 global_mean, global_std, transformer_depth, transformer_heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.neighbor = get_neighs_order(40962)
        self.parcel_index = Get_parcellation()
        self.num_patches = num_patches
        self.global_mean = global_mean
        self.global_std = global_std
        # token generator for time series objects
        num_ts_feats = 2  # original ts + its diff
        self.SpherResBlock = nn.Sequential(
            onering_conv_layer_batch(input_channel,hidden_dim,neigh_orders=self.neighbor),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            res_block(hidden_dim, hidden_dim, self.neighbor)
        )
        self.stem_layer = nn.Linear(153*6, hidden_dim)
        self.transformer = Transformer(hidden_dim, transformer_depth, transformer_heads, dim_head, mlp_dim, dropout)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-5)
            for i in range(num_ts_feats)
        ])

        # token generator for scalar statistics
        if scalar_scales is None:
            scalar_scales = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        num_scalar_stats = 6  # mean + std
        self.scalar_encoders = nn.ModuleList([
            MultiScaledScalarEncoder(
                scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc)
            for i in range(num_scalar_stats)
        ])

        # final token projector
        self.linear_encoder = LinearEncoder(
            hidden_dim_scalar_enc * num_scalar_stats + hidden_dim * (num_ts_feats), hidden_dim)

        # scales each time-series w.r.t. its mean and std
        # self.ts_scaler = lambda x: (x - torch.mean(x, axis=2, keepdim=True)) / (torch.std(x, axis=2, keepdim=True) + 1e-5)

    def forward(self, x):
        # x size is (B, C, Ns)
        # the input size is (B,C,P,V)
        with torch.no_grad():
            # compute statistics for each patch
            x_parcel = x[:,:,self.parcel_index]
            mean_patched = torch.mean(x_parcel, axis=-1, keepdim=True)
            std_patched = torch.std(x_parcel, axis=-1, keepdim=True)
            statistics = [mean_patched[:,0,:,:], mean_patched[:,1,:,:], mean_patched[:,2,:,:],std_patched[:,0,:,:],std_patched[:,1,:,:],std_patched[:,2,:,:]] # statistics=2*[B,C,P,1]


        # for each encoder output is (batch_size, num_sub_ts, hidden_dim_scalar_enc)
        scalar_embeddings = [self.scalar_encoders[i](
            statistics[i]) for i in range(len(statistics))] # (B,N,hidden_dim) * 6

        with torch.no_grad():
            x = (x - self.global_mean.to(device=x.device)) / self.global_std.to(device=x.device)
            x_parcel = x[:,:,self.parcel_index].transpose(-1,-2).contiguous().reshape([x.shape[0], -1, self.num_patches])

        # apply spherical cpnvolutions in vertex level
        vertex_embedding = self.SpherResBlock(x)

        # apply transformer in parcel level
        parcel_embedding = self.transformer(self.stem_layer(x_parcel.transpose(-1,-2).contiguous()))

        # apply convolution for original ts and its diff
        # concatenate diff_x, x, mu and std embeddinga and send them to the linear projector

        vertex_embedding = torch.mean(vertex_embedding[:,:,self.parcel_index],dim=-1).transpose(-1,-2).contiguous()
        x_embeddings = torch.cat([vertex_embedding,parcel_embedding, torch.cat(scalar_embeddings, dim=-1)], dim=-1)
        x_embeddings = self.linear_encoder(x_embeddings)
        return x_embeddings

class TokenGeneratorUnit_single(nn.Module):
    def __init__(self,  input_channel, hidden_dim, num_patches, scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc,
                 global_mean, global_std, transformer_depth, transformer_heads, dim_head, mlp_dim, dropout, TU_out_dim=512):
        super().__init__()
        self.neighbor = get_neighs_order(40962)
        self.parcel_index = Get_parcellation()
        self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962],axis=0)
        self.num_patches = num_patches
        # self.register_buffer("global_mean", global_mean)  # 不参与梯度
        # self.register_buffer("global_std", global_std)
        # token generator for time series objects
        num_ts_feats = 2  # original ts + its diff
        self.SpherResBlock = nn.Sequential(
            onering_conv_layer_batch(input_channel,hidden_dim,neigh_orders=self.neighbor),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            res_block(hidden_dim, hidden_dim, self.neighbor),
        )
        self.stem_layer = nn.Linear(153*input_channel, hidden_dim)
        self.transformer = Transformer(hidden_dim, transformer_depth, transformer_heads, dim_head, mlp_dim, dropout)

        # self.layer_norms = nn.ModuleList([
        #     nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-5)
        #     for i in range(num_ts_feats)
        # ])

        # token generator for scalar statistics
        if scalar_scales is None:
            scalar_scales = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        self.num_scalar_stats = input_channel*2  # mean + std
        self.scalar_encoders = nn.ModuleList([
            MultiScaledScalarEncoder(
                scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc)
            for i in range(self.num_scalar_stats)
        ])

        # final token projector
        self.linear_encoder = LinearEncoder(
            hidden_dim_scalar_enc * self.num_scalar_stats + hidden_dim * (num_ts_feats), TU_out_dim)

        # scales each time-series w.r.t. its mean and std
        # self.ts_scaler = lambda x: (x - torch.mean(x, axis=2, keepdim=True)) / (torch.std(x, axis=2, keepdim=True) + 1e-5)

    def forward(self, x):
        # x size is (B, C, Ns)
        # the input size is (B,C,P,V)
        with torch.no_grad():
            # compute statistics for each patch
            x_parcel = x[:,:,self.parcel_index]
            mean_patched = torch.mean(x_parcel, axis=-1, keepdim=True)
            std_patched = torch.std(x_parcel, axis=-1, keepdim=True)
            statistics = [mean_patched[:,i,:,:] for i in range(mean_patched.shape[1])] # statistics=2*[B,C,P,1]
            statistics_std = [std_patched[:,i,:,:] for i in range(std_patched.shape[1])]
            statistics.extend(statistics_std)
        # for each encoder output is (batch_size, num_sub_ts, hidden_dim_scalar_enc)
        scalar_embeddings = [self.scalar_encoders[i](
            statistics[i]) for i in range(len(statistics))] # (B,N,hidden_dim) * 6
        with torch.no_grad():
            # x = (x - self.global_mean[:,index,:]) / self.global_std[:,index,:]
            x_parcel = x[:,:,self.parcel_index].transpose(-1,-2).contiguous().reshape([x.shape[0], -1, self.num_patches])
        # apply spherical cpnvolutions in vertex level
        lh_vertex_embedding = self.SpherResBlock(x[:,:,:40962])
        rh_vertex_embedding = self.SpherResBlock(x[:,:,40962:])
        vertex_embedding = torch.cat([lh_vertex_embedding, rh_vertex_embedding], dim=-1)
        # apply transformer in parcel leve
        parcel_embedding = self.transformer(self.stem_layer(x_parcel.transpose(-1,-2).contiguous()))
        # concatenate diff_x, x, mu and std embeddinga and send them to the linear projector
        vertex_embedding = torch.mean(vertex_embedding[:,:,self.parcel_index],dim=-1).transpose(-1,-2).contiguous()
        x_embeddings = torch.cat([vertex_embedding,parcel_embedding, torch.cat(scalar_embeddings, dim=-1)], dim=-1)
        x_embeddings = self.linear_encoder(x_embeddings)

        return x_embeddings

class TokenGeneratorUnit_single_woSpherical(nn.Module):
    def __init__(self,  input_channel, hidden_dim, num_patches, scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc,
                 global_mean, global_std, transformer_depth, transformer_heads, dim_head, mlp_dim, dropout, TU_out_dim=512):
        super().__init__()
        self.neighbor = get_neighs_order(40962)
        self.parcel_index = Get_parcellation()
        self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962],axis=0)
        self.num_patches = num_patches
        # self.register_buffer("global_mean", global_mean)  # 不参与梯度
        # self.register_buffer("global_std", global_std)
        # token generator for time series objects
        num_ts_feats = 2  # original ts + its diff
        # self.SpherResBlock = nn.Sequential(
        #     onering_conv_layer_batch(input_channel,hidden_dim,neigh_orders=self.neighbor),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     res_block(hidden_dim, hidden_dim, self.neighbor),
        # )
        self.stem_layer = nn.Linear(153*input_channel, hidden_dim)
        self.transformer = Transformer(hidden_dim, transformer_depth, transformer_heads, dim_head, mlp_dim, dropout)

        # self.layer_norms = nn.ModuleList([
        #     nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-5)
        #     for i in range(num_ts_feats)
        # ])

        # token generator for scalar statistics
        if scalar_scales is None:
            scalar_scales = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        self.num_scalar_stats = input_channel*2  # mean + std
        self.scalar_encoders = nn.ModuleList([
            MultiScaledScalarEncoder(
                scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc)
            for i in range(self.num_scalar_stats)
        ])

        # final token projector
        self.linear_encoder = LinearEncoder(
            hidden_dim_scalar_enc * self.num_scalar_stats + hidden_dim, TU_out_dim)

        # scales each time-series w.r.t. its mean and std
        # self.ts_scaler = lambda x: (x - torch.mean(x, axis=2, keepdim=True)) / (torch.std(x, axis=2, keepdim=True) + 1e-5)

    def forward(self, x):
        # x size is (B, C, Ns)
        # the input size is (B,C,P,V)
        with torch.no_grad():
            # compute statistics for each patch
            x_parcel = x[:,:,self.parcel_index]
            mean_patched = torch.mean(x_parcel, axis=-1, keepdim=True)
            std_patched = torch.std(x_parcel, axis=-1, keepdim=True)
            statistics = [mean_patched[:,i,:,:] for i in range(mean_patched.shape[1])] # statistics=2*[B,C,P,1]
            statistics_std = [std_patched[:,i,:,:] for i in range(std_patched.shape[1])]
            statistics.extend(statistics_std)
        # for each encoder output is (batch_size, num_sub_ts, hidden_dim_scalar_enc)
        scalar_embeddings = [self.scalar_encoders[i](
            statistics[i]) for i in range(len(statistics))] # (B,N,hidden_dim) * 6
        with torch.no_grad():
            # x = (x - self.global_mean[:,index,:]) / self.global_std[:,index,:]
            x_parcel = x[:,:,self.parcel_index].transpose(-1,-2).contiguous().reshape([x.shape[0], -1, self.num_patches])
        # apply spherical cpnvolutions in vertex level
        # lh_vertex_embedding = self.SpherResBlock(x[:,:,:40962])
        # rh_vertex_embedding = self.SpherResBlock(x[:,:,40962:])
        # vertex_embedding = torch.cat([lh_vertex_embedding, rh_vertex_embedding], dim=-1)
        # apply transformer in parcel leve
        parcel_embedding = self.transformer(self.stem_layer(x_parcel.transpose(-1,-2).contiguous()))
        # concatenate diff_x, x, mu and std embeddinga and send them to the linear projector
        # vertex_embedding = torch.mean(vertex_embedding[:,:,self.parcel_index],dim=-1).transpose(-1,-2).contiguous()
        x_embeddings = torch.cat([parcel_embedding, torch.cat(scalar_embeddings, dim=-1)], dim=-1)
        x_embeddings = self.linear_encoder(x_embeddings)

        return x_embeddings

class TokenGeneratorUnit_single_woTransformer(nn.Module):
    def __init__(self,  input_channel, hidden_dim, num_patches, scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc,
                 global_mean, global_std, transformer_depth, transformer_heads, dim_head, mlp_dim, dropout, TU_out_dim=512):
        super().__init__()
        self.neighbor = get_neighs_order(40962)
        self.parcel_index = Get_parcellation()
        self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962],axis=0)
        self.num_patches = num_patches
        # self.register_buffer("global_mean", global_mean)  # 不参与梯度
        # self.register_buffer("global_std", global_std)
        # token generator for time series objects
        num_ts_feats = 2  # original ts + its diff
        self.SpherResBlock = nn.Sequential(
            onering_conv_layer_batch(input_channel,hidden_dim,neigh_orders=self.neighbor),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            res_block(hidden_dim, hidden_dim, self.neighbor),
        )
        # self.stem_layer = nn.Linear(153*input_channel, hidden_dim)
        # self.transformer = Transformer(hidden_dim, transformer_depth, transformer_heads, dim_head, mlp_dim, dropout)

        # self.layer_norms = nn.ModuleList([
        #     nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-5)
        #     for i in range(num_ts_feats)
        # ])

        # token generator for scalar statistics
        if scalar_scales is None:
            scalar_scales = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        self.num_scalar_stats = input_channel*2  # mean + std
        self.scalar_encoders = nn.ModuleList([
            MultiScaledScalarEncoder(
                scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc)
            for i in range(self.num_scalar_stats)
        ])

        # final token projector
        self.linear_encoder = LinearEncoder(
            hidden_dim_scalar_enc * self.num_scalar_stats + hidden_dim, TU_out_dim)

        # scales each time-series w.r.t. its mean and std
        # self.ts_scaler = lambda x: (x - torch.mean(x, axis=2, keepdim=True)) / (torch.std(x, axis=2, keepdim=True) + 1e-5)

    def forward(self, x):
        # x size is (B, C, Ns)
        # the input size is (B,C,P,V)
        with torch.no_grad():
            # compute statistics for each patch
            x_parcel = x[:,:,self.parcel_index]
            mean_patched = torch.mean(x_parcel, axis=-1, keepdim=True)
            std_patched = torch.std(x_parcel, axis=-1, keepdim=True)
            statistics = [mean_patched[:,i,:,:] for i in range(mean_patched.shape[1])] # statistics=2*[B,C,P,1]
            statistics_std = [std_patched[:,i,:,:] for i in range(std_patched.shape[1])]
            statistics.extend(statistics_std)
        # for each encoder output is (batch_size, num_sub_ts, hidden_dim_scalar_enc)
        scalar_embeddings = [self.scalar_encoders[i](
            statistics[i]) for i in range(len(statistics))] # (B,N,hidden_dim) * 6
        # with torch.no_grad():
        #     # x = (x - self.global_mean[:,index,:]) / self.global_std[:,index,:]
        #     x_parcel = x[:,:,self.parcel_index].transpose(-1,-2).contiguous().reshape([x.shape[0], -1, self.num_patches])
        # apply spherical cpnvolutions in vertex level
        lh_vertex_embedding = self.SpherResBlock(x[:,:,:40962])
        rh_vertex_embedding = self.SpherResBlock(x[:,:,40962:])
        vertex_embedding = torch.cat([lh_vertex_embedding, rh_vertex_embedding], dim=-1)
        # apply transformer in parcel leve
        # parcel_embedding = self.transformer(self.stem_layer(x_parcel.transpose(-1,-2).contiguous()))
        # concatenate diff_x, x, mu and std embeddinga and send them to the linear projector
        vertex_embedding = torch.mean(vertex_embedding[:,:,self.parcel_index],dim=-1).transpose(-1,-2).contiguous()
        x_embeddings = torch.cat([vertex_embedding, torch.cat(scalar_embeddings, dim=-1)], dim=-1)
        x_embeddings = self.linear_encoder(x_embeddings)

        return x_embeddings

class TokenGeneratorUnit_single_woScalar(nn.Module):
    def __init__(self,  input_channel, hidden_dim, num_patches, scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc,
                 global_mean, global_std, transformer_depth, transformer_heads, dim_head, mlp_dim, dropout, TU_out_dim=512):
        super().__init__()
        self.neighbor = get_neighs_order(40962)
        self.parcel_index = Get_parcellation()
        self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962],axis=0)
        self.num_patches = num_patches
        # self.register_buffer("global_mean", global_mean)  # 不参与梯度
        # self.register_buffer("global_std", global_std)
        # token generator for time series objects
        num_ts_feats = 2  # original ts + its diff
        self.SpherResBlock = nn.Sequential(
            onering_conv_layer_batch(input_channel,hidden_dim,neigh_orders=self.neighbor),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            res_block(hidden_dim, hidden_dim, self.neighbor),
        )
        self.stem_layer = nn.Linear(153*input_channel, hidden_dim)
        self.transformer = Transformer(hidden_dim, transformer_depth, transformer_heads, dim_head, mlp_dim, dropout)

        # self.layer_norms = nn.ModuleList([
        #     nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-5)
        #     for i in range(num_ts_feats)
        # ])

        # token generator for scalar statistics
        if scalar_scales is None:
            scalar_scales = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        self.num_scalar_stats = input_channel*2  # mean + std
        # self.scalar_encoders = nn.ModuleList([
        #     MultiScaledScalarEncoder(
        #         scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc)
        #     for i in range(self.num_scalar_stats)
        # ])

        # final token projector
        self.linear_encoder = LinearEncoder(hidden_dim * num_ts_feats, TU_out_dim)

        # scales each time-series w.r.t. its mean and std
        # self.ts_scaler = lambda x: (x - torch.mean(x, axis=2, keepdim=True)) / (torch.std(x, axis=2, keepdim=True) + 1e-5)

    def forward(self, x):
        # x size is (B, C, Ns)
        # the input size is (B,C,P,V)
        with torch.no_grad():
            # compute statistics for each patch
            x_parcel = x[:,:,self.parcel_index]
            mean_patched = torch.mean(x_parcel, axis=-1, keepdim=True)
            std_patched = torch.std(x_parcel, axis=-1, keepdim=True)
            statistics = [mean_patched[:,i,:,:] for i in range(mean_patched.shape[1])] # statistics=2*[B,C,P,1]
            statistics_std = [std_patched[:,i,:,:] for i in range(std_patched.shape[1])]
            statistics.extend(statistics_std)
        # for each encoder output is (batch_size, num_sub_ts, hidden_dim_scalar_enc)
        # scalar_embeddings = [self.scalar_encoders[i](
        #     statistics[i]) for i in range(len(statistics))] # (B,N,hidden_dim) * 6
        with torch.no_grad():
            # x = (x - self.global_mean[:,index,:]) / self.global_std[:,index,:]
            x_parcel = x[:,:,self.parcel_index].transpose(-1,-2).contiguous().reshape([x.shape[0], -1, self.num_patches])
        # apply spherical cpnvolutions in vertex level
        lh_vertex_embedding = self.SpherResBlock(x[:,:,:40962])
        rh_vertex_embedding = self.SpherResBlock(x[:,:,40962:])
        vertex_embedding = torch.cat([lh_vertex_embedding, rh_vertex_embedding], dim=-1)
        # apply transformer in parcel leve
        parcel_embedding = self.transformer(self.stem_layer(x_parcel.transpose(-1,-2).contiguous()))
        # concatenate diff_x, x, mu and std embeddinga and send them to the linear projector
        vertex_embedding = torch.mean(vertex_embedding[:,:,self.parcel_index],dim=-1).transpose(-1,-2).contiguous()
        x_embeddings = torch.cat([vertex_embedding, parcel_embedding], dim=-1)
        x_embeddings = self.linear_encoder(x_embeddings)

        return x_embeddings


class ViTUnit(nn.Module):
    def __init__(self, hidden_dim, num_patches, depth, heads, mlp_dim, dim_head, dropout, device):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim, dropout=dropout, max_len=num_patches + 1)
        self.cls_token = nn.Parameter(torch.randn(hidden_dim).to(device))
        self.transformer = Transformer(
            hidden_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x_embeddings, ps = pack([cls_tokens, x], 'b * d')
        x_embeddings = self.pos_encoder(
            x_embeddings.transpose(0, 1)).transpose(0, 1)
        x_embeddings = self.transformer(x_embeddings)
        cls_tokens, _ = unpack(x_embeddings, ps, 'b * d')
        return cls_tokens.reshape(cls_tokens.shape[0], -1)

class SurfEncoder(nn.Module):
    def __init__(self, input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                 global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout,
                 Vit_depth, Vit_heads, Vit_mlp_dim, Vit_dim_head, Vit_dropout, pretrain=False):
        super().__init__()
        self.tokenizer = TokenGeneratorUnit(input_channel,hidden_dim,num_patches, TU_scalar_scales,TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                 global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout)
        self.vit = ViTUnit(hidden_dim, num_patches, Vit_depth, Vit_heads, Vit_mlp_dim, Vit_dim_head, Vit_dropout, "cpu")
        self.pretrain = pretrain
        self.prj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        tokens = self.tokenizer(x)
        vit = self.vit(tokens)
        if self.pretrain:
            return self.prj(vit)
        else:
            return vit

# class Mantis8M(
#     nn.Module,
#     PyTorchModelHubMixin,
#     # optionally, you can add metadata which gets pushed to the model card
#     library_name="mantis",
#     repo_url="https://huggingface.co/paris-noah/Mantis-8M/tree/main",
#     pipeline_tag="time-series-foundation-model",
#     license="mit",
#     tags=["time-series-foundation-model"],
# ):
#     """
#     The architecture of Mantis time series foundation model.
#
#     Parameters
#     ----------
#     seq_len: int, default 512
#         The sequence length, i.e., the length of each time series. This model does not support data with non-fixed
#         sequence length, please make all the time series to be of a fixed length by resizing or padding.
#     hidden_dim: int, default=256
#         Size of a patch (token), i.e., what the hidden dimension each patch is projected to. At the same time,
#         ``hidden_dim`` corresponds to the dimension of the embedding space.
#     num_patches: int, default=32
#         Number of patches (tokens).
#     scalar_scales: list, default=None
#         List of scales used for MultiScaledScalarEncoder in TokenGeneratorUnit. By default, initialized as [1e-4, 1e-3,
#         1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4].
#     hidden_dim_scalar_enc: int, default=32
#         Hidden dimension of a scalar encoder used for MultiScaledScalarEncoder in TokenGeneratorUnit.
#     epsilon_scalar_enc: float, default=1.1
#         A constant term used to tolerate the computational error in computation of scale weights for
#         MultiScaledScalarEncoder in TokenGeneratorUnit.
#     transf_depth: int, default=6
#         Number of transformer layers used for Transformer in ViTUnit.
#     transf_num_heads: int, default=8
#         Number of self-attention heads used for Transformer in ViTUnit.
#     transf_mlp_dim: int, default=512
#         Hidden dimension of the MLP (feed-forward) transformer's part used for Transformer in ViTUnit.
#     transf_dim_head: int, default=128
#         Hidden dimension of the keys, queries and values used for Transformer in ViTUnit.
#     transf_dropout: froat, default=0.1
#         Dropout value used for Transformer in ViTUnit.
#     device: {'cpu', 'cuda'}, default='cuda'
#         On which device the model is located.
#     pre_training: bool, default=False
#         If True, applies an MLP projector after the ViTUnit, which originally was used to pre-train the model using
#         InfoNCE contrastive loss.
#     """
#
#     def __init__(self, seq_len=512, hidden_dim=256, num_patches=32, scalar_scales=None, hidden_dim_scalar_enc=32,
#                  epsilon_scalar_enc=1.1, transf_depth=6, transf_num_heads=8, transf_mlp_dim=512, transf_dim_head=128,
#                  transf_dropout=0.1, device='cuda', pre_training=False):
#
#         super().__init__()
#         assert (seq_len % num_patches) == 0, print(
#             'Seq_len must be the multiple of num_patches')
#         patch_window_size = int(seq_len / num_patches)
#
#         self.hidden_dim = hidden_dim
#         self.num_patches = num_patches
#         self.scalar_scales = scalar_scales
#         self.hidden_dim_scalar_enc = hidden_dim_scalar_enc
#         self.epsilon_scalar_enc = epsilon_scalar_enc
#         self.seq_len = seq_len
#         self.pre_training = pre_training
#
#         self.tokgen_unit = TokenGeneratorUnit(hidden_dim=hidden_dim,
#                                               num_patches=num_patches,
#                                               patch_window_size=patch_window_size,
#                                               scalar_scales=scalar_scales,
#                                               hidden_dim_scalar_enc=hidden_dim_scalar_enc,
#                                               epsilon_scalar_enc=epsilon_scalar_enc)
#         self.vit_unit = ViTUnit(hidden_dim=hidden_dim, num_patches=num_patches, depth=transf_depth,
#                                 heads=transf_num_heads, mlp_dim=transf_mlp_dim, dim_head=transf_dim_head,
#                                 dropout=transf_dropout, device=device)
#
#         self.prj = nn.Sequential(
#             nn.LayerNorm(self.hidden_dim),
#             nn.Linear(self.hidden_dim, self.hidden_dim)
#         )
#
#         self.to(device)
#
#     def to(self, device):
#         self.device = device
#         return super().to(device)
#
#     def forward(self, x):
#         x_embeddings = self.tokgen_unit(x)
#         vit_out = self.vit_unit(x_embeddings)
#         if self.pre_training:
#             return self.prj(vit_out)
#         else:
#             return vit_out

if __name__ == '__main__':
    model = SurfEncoder(3, 512, 320, None, 32,1.1, 2,1,
                               2,4,128,512,0.1, 6,8,512,128,0.1,True)
    temp = torch.randn(1, 3, 40962)
    from calflops import calculate_flops

    flops, macs, params = calculate_flops(model, input_shape=(1, 3, 40962))
    print(flops)
    print(macs)
    print(params)


    print(model(temp).shape)
