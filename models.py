
# Based on CLIP code bases
# Modified from github.com/openai/CLIP
# --------------------------------------------------------'

from collections import OrderedDict
import numpy as np
import timm
import torch
from torch import nn
import torchvision.transforms._functional_tensor as F_t
from functools import partial

from timm.models.vision_transformer import VisionTransformer, Block
from timm.models.registry import register_model
from timm.models.vision_transformer import (
    default_cfgs,
    build_model_with_cfg,
    checkpoint_filter_fn,
)

from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from util.utils import Get_parcellation
from utils import get_2d_sincos_pos_embed, iBOTHead, get_1d_sincos_pos_embed_from_grid


# def get_att_mask(attention, ratio=0.5):
#     bs = attention.shape[0]
#     masks = torch.ones((bs,49), dtype=torch.bool, device=attention.device)
#     attention = attention.reshape((-1, 14, 14))
#     attention = torch.nn.functional.interpolate(attention.unsqueeze(1), (7, 7), mode='bilinear').squeeze()
#     attention = attention.reshape(bs,-1)
#     N = int(attention.shape[1] * ratio)
#
#     reservation = torch.argsort(attention, descending=True)
#     reservation = reservation[:,:N+1] # get top N values
#     masks = masks.scatter_(1, reservation, False)
#
#     full_mask = torch.zeros((bs, 14, 14), dtype=torch.bool, device=attention.device)
#     full_mask[:, 0::2, 0::2] = masks.reshape(bs, 7, 7)
#     full_mask[:, 0::2, 1::2] = masks.reshape(bs, 7, 7)
#     full_mask[:, 1::2, 0::2] = masks.reshape(bs, 7, 7)
#     full_mask[:, 1::2, 1::2] = masks.reshape(bs, 7, 7)
#     full_mask = full_mask.reshape(bs, -1)
#
#     return full_mask

def get_att_mask(attention, ratio=0.5):
    # attention size = (B, N)
    bs = attention.shape[0]
    masks = torch.ones((attention.shape), dtype=torch.bool, device=attention.device)
    N = int(attention.shape[1] * ratio)
    reservation = torch.argsort(attention, descending=True)
    reservation = reservation[:, :N + 1]  # get top N values
    masks = masks.scatter_(1, reservation, False)
    return masks



def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0] # value of x after applying multi-headed self attention

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        vision_width: int,
        vision_model: nn.Module,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        **kwargs,
    ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model

        self.transformer = Transformer(
            width=transformer_width, # im assuming this is embedding size of text transformer
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(), # auto-regressive mask for text transformer.
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width) # [vocab_size, transformer_width]
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim)) # final projection before CLIP comparison
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02) # each weight is independently and randomly sampled from normal dist of specified std
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        ) # the product of 1/sqrt(transformer width) and 1/sqrt(2* number of transformer layers)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std) # in_proj couples Q,K, and V. So, we initialize them with the same std.
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std) # out_proj is the linear
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width**-0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image):
        x,_,_,_ = self.visual(image)
        # print(x.shape)
        x_o = x[:,0,:] @ self.image_projection
        # print(x_o.shape)
        return x_o

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "logit_scale": self.logit_scale.exp(),
        }


class Proj_Head(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size, bias=False),
            LayerNorm(mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size, bias=False),
            LayerNorm(mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, projection_size, bias=False),
            LayerNorm(projection_size),
        )

    def forward(self, x):
        return self.net(x)


class Pred_Head(nn.Module):
    def __init__(self, dim, mlp_hidden_size, projection_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, mlp_hidden_size),
            LayerNorm(mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        """ page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        """
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(
        self, in_dim=2048, hidden_dim=512, out_dim=2048
    ):  # bottleneck structure
        super().__init__()
        """ page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        """
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ACLIP(CLIP):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_ratio = kwargs["mask_ratio"]
        self.visual_ema = kwargs['vision_model_ema']
        vision_width = kwargs["vision_width"]
        embed_dim = kwargs['embed_dim']
    
        self.image_mlp = self._build_mlp(vision_width, 4096, 256) #for simclr
        self.im_proj_byol = self._build_mlp_byol(2, vision_width, 4096, 256, False)
        self.im_pred_byol = self._build_mlp_byol(2, 256, 4096, 256, False)

        transformer_width = kwargs['transformer_width']
        transformer_layers = kwargs['transformer_layers']
        transformer_heads = kwargs['transformer_heads']

        # text ema
        self.transformer_e = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.image_projection_e = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection_e = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.im_proj_byol_e = self._build_mlp_byol(2, vision_width, 4096, 256, False)
        
        for param_m, param_b in zip(self.visual_ema.parameters(), self.visual.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        for param_m, param_b in zip(self.transformer_e.parameters(), self.transformer.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        for param_m, param_b in zip(self.im_proj_byol_e.parameters(), self.im_proj_byol.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.image_projection_e.requires_grad = False 
        self.image_projection_e.data.copy_(self.image_projection.data)
        self.text_projection_e.requires_grad = False
        self.text_projection_e.data.copy_(self.text_projection.data)


    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(
            self.visual.parameters(), self.visual_ema.parameters()
        ):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)

        for param_b, param_m in zip(
            self.transformer.parameters(), self.transformer_e.parameters()
        ):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)
        for param_b, param_m in zip(
            self.im_proj_byol.parameters(), self.im_proj_byol_e.parameters()
        ):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)

        self.image_projection_e.data = self.image_projection_e.data * m + self.image_projection * (1.0 - m)
        self.text_projection_e.data = self.text_projection_e.data * m + self.text_projection * (1.0 - m)        

    def _build_mlp_byol(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    # from simclr
    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(
            OrderedDict(
                [
                    ("layer1", nn.Linear(in_dim, mlp_dim)),
                    ("bn1", nn.BatchNorm1d(mlp_dim)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("layer2", nn.Linear(mlp_dim, mlp_dim)),
                    ("bn2", nn.BatchNorm1d(mlp_dim)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("layer3", nn.Linear(mlp_dim, out_dim)),
                ]
            )
        )

    def encode_image(self, image, mask=None, ret=False, ema=False):
        if ema == False:
            x, attn, _, _= self.visual(image, mask=mask, need_attn=False)
            tokens = x
            x = x[:, 0] @ self.image_projection
        else:
            x, attn, _, _ = self.visual_ema(image, mask=mask, need_attn=True)
            tokens = x
            x = x[:, 0] @ self.image_projection_e

        if ret:
            return x, attn, tokens
        return x

    def encode_text(self, text, ema=False):
        if ema:
            text_projection = self.text_projection_e
            transformer = self.transformer_e
        else:
            text_projection = self.text_projection
            transformer = self.transformer

        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ text_projection

        return x
    
    def get_mask(self,attention, ratio=0.5):
        # attention size = (B, N)
        bs = attention.shape[0]
        masks = torch.ones((attention.shape), dtype=torch.bool, device=attention.device)
        N = int(attention.shape[1] * ratio)
        reservation = torch.argsort(attention, descending=True)
        reservation = reservation[:, :N + 1]  # get top N values
        masks = masks.scatter_(1, reservation, False)
        return masks

    def forward(self, img, text, momentum):
        # im1 is actually the concat of the two augmented images. im2 is the larger cropped version without augmentation.
        im1 = img[:,0,:,:]
        im2 = img[:,1,:,:]
        with torch.no_grad():
            self._update_momentum_encoder(momentum)
            x, attn, _, _ = self.visual_ema(im2, need_attn=True) # x is forward pass from the teacher with embeddings for all 197 patches, attn is the attention map of CLS token
            cls_token = x[:, 0] # CLS token embedding from the teacher
            byol_feats_e = self.im_proj_byol_e(cls_token) # (batch_size, 256) output

        attention_map = attn
        mask = get_att_mask(attention_map, ratio=self.mask_ratio)
        def sample_mask_independent(mask, frac=0.5):
            """
            独立采样: 每次对 True 部分采样一部分 (可能有重叠)
            :param mask: (B, L) bool
            :param frac: 每次采样比例
            :return: (B, L) bool
            """
            B, L = mask.shape
            sampled_mask = torch.zeros_like(mask, dtype=torch.bool)
            for i in range(B):
                true_idx = torch.where(mask[i])[0]
                if len(true_idx) == 0:
                    continue
                k = max(1, int(len(true_idx) * frac))
                chosen = true_idx[torch.randperm(len(true_idx))[:k]]
                sampled_mask[i, chosen] = True
            return sampled_mask

        # 使用
        mask1 = sample_mask_independent(mask, frac=0.8)
        mask2 = sample_mask_independent(mask, frac=0.8)
        image_embed, _, tokens = self.encode_image(im1, mask=mask1, ret=True)
        image_embed_1, _, tokens_1 = self.encode_image(im1, mask=mask2, ret=True)
        bs = text.shape[0]
        cls_token = torch.concatenate([tokens[:, 0], tokens_1[:,0]],dim=0)
        image_embed = torch.concatenate([image_embed, image_embed_1],dim=0)
        image_ssl_embed = self.image_mlp(cls_token) # 256 dimensions with bs*2
        byol_feats = self.im_proj_byol(cls_token) # 256 dimensions with bs*2
        byol_feats = self.im_pred_byol(byol_feats) # 256 dimensions with bs*2, not sure why there are two levels for this
        text_embed = self.encode_text(text) # 512 dimension for CLIP task
        return {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "image_ssl_embed": image_ssl_embed,
            "byol_feats": byol_feats,
            "byol_feats_e": byol_feats_e,
            "logit_scale": self.logit_scale.exp(),
        }

class Finetuning_ACLIP(nn.Module):
    def __init__(self, hidden_dim, visual, output_channel=2):
        super(Finetuning_ACLIP, self).__init__()
        self.visual = visual

        self.linear_head = nn.Sequential(
            nn.Linear(hidden_dim, output_channel),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(0.2),
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(0.2),
            # nn.Linear(128, output_channel),
        )
        self.initiate_parameters()
        self.parcel_index = Get_parcellation()
        self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962], axis=0)


    def encode_image(self, image, mask=None, ret=False, ema=False):
        if ema == False:
            x, attn, ids_restore, mask = self.visual(image, mask=mask)
            tokens = x
            x = x[:, 0]
        else:
            x, attn, ids_restore, mask = self.visual_ema(image, mask=mask)
            tokens = x
            x = x[:, 0]

        if ret:
            return x, attn, ids_restore, tokens, mask
        return x


    def initiate_parameters(self):
        for name, module in self.linear_head.named_children():
            # 只对Linear层进行初始化
            if isinstance(module, nn.Linear):
                # 初始化权重
                nn.init.xavier_uniform_(module.weight)
                # 初始化偏置 (统一设置为0.1)
                nn.init.constant_(module.bias, 0.1)


    def forward(self, u):
        x_parcel = u[:, :, self.parcel_index]
        u = x_parcel.transpose(-1, -2).contiguous().reshape([x_parcel.shape[0], -1, 640])
        img_embed_u = self.encode_image(u)
        cls = self.linear_head(img_embed_u)
        return {
            "img_embed_u": img_embed_u,
            "cls": cls
        }

class Finetuning_DetailCLIP(nn.Module):
    def __init__(self, hidden_dim, visual, output_channel=2):
        super(Finetuning_DetailCLIP, self).__init__()
        self.visual = visual

        self.linear_head = nn.Sequential(
            nn.Linear(hidden_dim, output_channel),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(0.2),
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(0.2),
            # nn.Linear(128, output_channel),
        )
        self.initiate_parameters()
        self.parcel_index = Get_parcellation()
        self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962], axis=0)


    def encode_image(self, image, mask=None, ret=False, ema=False):
        if ema == False:
            x, attn, ids_restore, mask = self.visual(image, mask=mask)
            tokens = x
            x = x[:, 0]
        else:
            x, attn, ids_restore, mask = self.visual_ema(image, mask=mask)
            tokens = x
            x = x[:, 0]

        if ret:
            return x, attn, ids_restore, tokens, mask
        return x


    def initiate_parameters(self):
        for name, module in self.linear_head.named_children():
            # 只对Linear层进行初始化
            if isinstance(module, nn.Linear):
                # 初始化权重
                nn.init.xavier_uniform_(module.weight)
                # 初始化偏置 (统一设置为0.1)
                nn.init.constant_(module.bias, 0.1)


    def forward(self, u):
        x_parcel = u[:, :, self.parcel_index]
        u = x_parcel.transpose(-1, -2).contiguous().reshape([x_parcel.shape[0], -1, 640])
        img_embed_u = self.encode_image(u)
        cls = self.linear_head(img_embed_u)
        return {
            "img_embed_u": img_embed_u,
            "cls": cls
        }

class Finetuning_parcellation_CLIP(nn.Module):
    def __init__(self, hidden_dim, visual, output_channel=2):
        super(Finetuning_parcellation_CLIP, self).__init__()
        self.visual = visual

        # self.linear_head = nn.Sequential(
        #     nn.Linear(hidden_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(128, output_channel),
        # )
        # self.initiate_parameters()
        # self.parcel_index = Get_parcellation()
        # self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962], axis=0)
        self.linear_head = nn.Linear(hidden_dim, 128 * 153, bias=True)
        self.predictor = nn.Conv1d(128, 36, kernel_size=1, bias=True)
        self.parcel_index_1 = Get_parcellation()
        self.parcel_index = torch.tensor(
            np.concatenate([self.parcel_index_1, self.parcel_index_1 + 40962], axis=0)).reshape([-1]).long()


    def encode_image(self, image, mask=None, ret=False, ema=False):
        if ema == False:
            x, attn, ids_restore, mask = self.visual(image, mask=mask)
            tokens = x
            x = x[:, 0]
        else:
            x, attn, ids_restore, mask = self.visual_ema(image, mask=mask)
            tokens = x
            x = x[:, 0]

        if ret:
            return x, attn, ids_restore, tokens, mask
        return x


    def initiate_parameters(self):
        for name, module in self.linear_head.named_children():
            # 只对Linear层进行初始化
            if isinstance(module, nn.Linear):
                # 初始化权重
                nn.init.xavier_uniform_(module.weight)
                # 初始化偏置 (统一设置为0.1)
                nn.init.constant_(module.bias, 0.1)


    def forward(self, u):
        x_parcel = u[:, :, self.parcel_index]
        u = x_parcel.transpose(-1, -2).contiguous().reshape([x_parcel.shape[0], -1, 640])
        x, attn, ids_restore, tokens, mask = self.encode_image(u, ret=True)
        x_recons = self.linear_head(tokens[:, 1:, :])
        out_sum = torch.zeros(x_recons.shape[0], 128, 81924, dtype=x_recons.dtype, device=x_recons.device)
        x_recons = x_recons.reshape(x_recons.shape[0], 640, 128, 153).permute(0, 2, 1, 3).reshape(
            [x_recons.shape[0], 128, -1])
        self.parcel_index = self.parcel_index.to(x_recons.device)
        out_sum.scatter_add_(dim=2,
                             index=self.parcel_index.unsqueeze(0).unsqueeze(0).expand(x_recons.shape[0], 128, -1),
                             src=x_recons)
        counts = torch.bincount(self.parcel_index, minlength=81924)
        out = out_sum / counts
        out = self.predictor(out)
        return {
            "img_embed_u": x,
            "cls": out
        }

class Finetuning_CLIP(nn.Module):
    def __init__(self, hidden_dim, visual, output_channel=2):
        super(Finetuning_CLIP, self).__init__()
        self.visual = visual

        self.linear_head = nn.Sequential(
            nn.Linear(hidden_dim, output_channel),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(0.2),
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(0.2),
            # nn.Linear(128, output_channel),
        )
        self.initiate_parameters()
        self.parcel_index = Get_parcellation()
        self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962], axis=0)


    def encode_image(self, image, mask=None, ret=False, ema=False):
        if ema == False:
            x, attn, ids_restore, mask = self.visual(image, mask=mask)
            tokens = x
            x = x[:, 0]
        else:
            x, attn, ids_restore, mask = self.visual_ema(image, mask=mask)
            tokens = x
            x = x[:, 0]

        if ret:
            return x, attn, ids_restore, tokens, mask
        return x


    def initiate_parameters(self):
        for name, module in self.linear_head.named_children():
            # 只对Linear层进行初始化
            if isinstance(module, nn.Linear):
                # 初始化权重
                nn.init.xavier_uniform_(module.weight)
                # 初始化偏置 (统一设置为0.1)
                nn.init.constant_(module.bias, 0.1)


    def forward(self, u):
        x_parcel = u[:, :, self.parcel_index]
        u= x_parcel.transpose(-1,-2).contiguous().reshape([x_parcel.shape[0],-1,640])
        img_embed_u = self.encode_image(u)
        cls = self.linear_head(img_embed_u)
        return {
            "img_embed_u": img_embed_u,
            "cls": cls
        }

class DetailCLIP(CLIP):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_ratio = kwargs["mask_ratio"]
        self.visual_ema = kwargs['vision_model_ema']
        vision_width = kwargs["vision_width"]
        embed_dim = kwargs['embed_dim']

        transformer_width = kwargs['transformer_width']
        transformer_layers = kwargs['transformer_layers']
        transformer_heads = kwargs['transformer_heads']
        # text ema
        self.transformer_e = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.image_projection_e = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection_e = nn.Parameter(torch.empty(transformer_width, embed_dim))

        for param_m, param_b in zip(self.visual_ema.parameters(), self.visual.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        for param_m, param_b in zip(self.transformer_e.parameters(), self.transformer.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient


        self.image_projection_e.requires_grad = False 
        self.image_projection_e.data.copy_(self.image_projection.data)
        self.text_projection_e.requires_grad = False
        self.text_projection_e.data.copy_(self.text_projection.data)

        # --------------------------------------------------------------------------
        # decoder specifics
        decoder_depth = kwargs['decoder_depth']
        decoder_num_heads = kwargs['decoder_num_heads']
        mlp_ratio = kwargs['mlp_ratio']
        norm_layer = kwargs['norm_layer']
        self.decoder_embed = nn.Linear(vision_width, vision_width, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, vision_width))
        num_patches = self.visual.patch_embed.num_patches

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, vision_width), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(vision_width, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(vision_width)
        self.initialize_decoder()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # projection heads
        # MAE projection head
        print('\tCreating MAE projection head')
        patch_size = kwargs['patch_size']
        in_chans = kwargs['in_chans']
        # self.reconstruction_pred = nn.Linear(vision_width, patch_size**2 * in_chans, bias = True) # MAE loss
        self.reconstruction_pred = nn.Linear(vision_width, in_chans*153, bias=True)  # MSE loss
        print('\tMAE projection head created')

        # IBOT projection head
        print('\tCreating IBOT projection head')
        out_dim = kwargs['out_dim']
        patch_out_dim = kwargs['patch_out_dim']
        norm_in_head = kwargs['norm_in_head']
        act_in_head = kwargs['act_in_head']
        shared_head_teacher = kwargs['shared_head_teacher']
        self.ibot_head = iBOTHead(
            vision_width, 
            out_dim,
            patch_out_dim=patch_out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
        )
        self.ibot_head_e = iBOTHead(
            vision_width, 
            out_dim,
            patch_out_dim=patch_out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
        )
        result = self.ibot_head_e.load_state_dict(self.ibot_head.state_dict(), strict=False)
        print('\tkeys have been loaded for ibot head with status:', result)
        print('\tIBOT projection head created')
        for p in self.ibot_head_e.parameters():
            p.requires_grad = False
        # --------------------------------------------------------------------------
        print('\tDetailCLIP model created')
        


    # def initialize_decoder(self):
    #     torch.nn.init.normal_(self.mask_token, std=.02)
    #     torch.nn.init.xavier_uniform_(self.decoder_embed.weight)
    #     decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.visual.patch_embed.num_patches**.5), cls_token=True)
    #     self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
    #
    #     def _init_weights_for_block(m):
    #         """Apply custom initialization to each module in the block."""
    #         if isinstance(m, nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)
    #
    #     for block in self.decoder_blocks:
    #         block.apply(_init_weights_for_block)
    #
    #     nn.init.constant_(self.decoder_norm.bias, 0)
    #     nn.init.constant_(self.decoder_norm.weight, 1.0)
    def initialize_decoder(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.xavier_uniform_(self.decoder_embed.weight)
        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_pos_embed.shape[-1],
                                                    np.arange(self.visual.patch_embed.num_patches), class_token = True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        def _init_weights_for_block(m):
            """Apply custom initialization to each module in the block."""
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        for block in self.decoder_blocks:
            block.apply(_init_weights_for_block)

        nn.init.constant_(self.decoder_norm.bias, 0)
        nn.init.constant_(self.decoder_norm.weight, 1.0)


    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(
            self.visual.parameters(), self.visual_ema.parameters()
        ):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)

        for param_b, param_m in zip(
            self.transformer.parameters(), self.transformer_e.parameters()
        ):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)
        for param_b, param_m in zip(
            self.ibot_head.parameters(), self.ibot_head_e.parameters()
        ):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)

        self.image_projection_e.data = self.image_projection_e.data * m + self.image_projection * (1.0 - m)
        self.text_projection_e.data = self.text_projection_e.data * m + self.text_projection * (1.0 - m)    


    def encode_image(self, image, mask=None, ret=False, ema=False):
        if ema == False:
            x, attn, ids_restore, mask = self.visual(image, mask=mask)
            tokens = x
            x = x[:, 0] @ self.image_projection
        else:
            x, attn, ids_restore, mask = self.visual_ema(image, mask=mask)
            tokens = x
            x = x[:, 0] @ self.image_projection_e

        if ret:
            return x, attn, ids_restore, tokens, mask
        return x

    def encode_text(self, text, ema=False):
        if ema:
            text_projection = self.text_projection_e
            transformer = self.transformer_e
        else:
            text_projection = self.text_projection
            transformer = self.transformer

        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ text_projection
        return x

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        return x
        
    def forward(self, img, text, momentum):
        # u, v are two augmented images
        u = img[:, 0, :, :]
        v = img[:, 1, :, :]
        with torch.no_grad():
            self._update_momentum_encoder(momentum)
            u_e, attn_u, _, _= self.visual_ema(u, need_attn=True)
            v_e, attn_v, _, _ = self.visual_ema(v, need_attn=True)
            teacher_ibot = self.ibot_head_e(torch.cat([u_e,v_e],dim=0))

        # obtain masks
        mask_u = get_att_mask(attn_u, ratio=self.mask_ratio)
        mask_v = get_att_mask(attn_v, ratio=self.mask_ratio)

        img_embed_u, _, ids_restore_u, latent_u, _ = self.encode_image(u, mask=mask_u, ret=True)
        img_embed_v, _, ids_restore_v, latent_v, _ = self.encode_image(v, mask=mask_v, ret=True)
        u_s = self.forward_decoder(latent_u, ids_restore_u)
        u_s_reconstructed = self.reconstruction_pred(u_s)[:, 1:, :]
        v_s = self.forward_decoder(latent_v, ids_restore_v)
        v_s_reconstructed = self.reconstruction_pred(v_s)[:, 1:, :]

        u_s[:, 0] = latent_u[:, 0] # assigning the cls token of u_s to latent_u's cls token to keep the cls token same for ibot task
        v_s[:, 0] = latent_v[:, 0] # assigning the cls token of v_s to latent_v's cls token to keep the cls token same for ibot task
        student_ibot = self.ibot_head(torch.cat([u_s,v_s],dim=0))
        text_embed = self.encode_text(text)
        return {
            # images
            "u": u,
            "v": v,
            
            # CLIP outputs
            "img_embed_u": img_embed_u,
            "img_embed_v": img_embed_v,
            "text_embed": text_embed,
            "logit_scale": self.logit_scale.exp(),

            # IBOT outputs
            "teacher_ibot": teacher_ibot,
            "student_ibot": student_ibot,

            # MAE outputs
            "u_s_reconstructed": u_s_reconstructed,
            "v_s_reconstructed": v_s_reconstructed,
            
            # masks
            "mask_u": mask_u,
            "mask_v": mask_v,
        }


class Finetuning_parcellation_DetailCLIP(nn.Module):
    def __init__(self, hidden_dim, visual, output_channel=2):
        super(Finetuning_parcellation_DetailCLIP, self).__init__()
        self.visual = visual

        # self.linear_head = nn.Sequential(
        #     nn.Linear(hidden_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(128, output_channel),
        # )
        # self.initiate_parameters()
        # self.parcel_index = Get_parcellation()
        # self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962], axis=0)
        self.linear_head = nn.Linear(hidden_dim, 128 * 153, bias=True)
        self.predictor = nn.Conv1d(128, 36, kernel_size=1, bias=True)
        self.parcel_index_1 = Get_parcellation()
        self.parcel_index = torch.tensor(
            np.concatenate([self.parcel_index_1, self.parcel_index_1 + 40962], axis=0)).reshape([-1]).long()


    def encode_image(self, image, mask=None, ret=False, ema=False):
        if ema == False:
            x, attn, ids_restore, mask = self.visual(image, mask=mask)
            tokens = x
            x = x[:, 0]
        else:
            x, attn, ids_restore, mask = self.visual_ema(image, mask=mask)
            tokens = x
            x = x[:, 0]

        if ret:
            return x, attn, ids_restore, tokens, mask
        return x


    def initiate_parameters(self):
        for name, module in self.linear_head.named_children():
            # 只对Linear层进行初始化
            if isinstance(module, nn.Linear):
                # 初始化权重
                nn.init.xavier_uniform_(module.weight)
                # 初始化偏置 (统一设置为0.1)
                nn.init.constant_(module.bias, 0.1)


    def forward(self, u):
        x_parcel = u[:, :, self.parcel_index]
        u = x_parcel.transpose(-1, -2).contiguous().reshape([x_parcel.shape[0], -1, 640])
        x, attn, ids_restore, tokens, mask = self.encode_image(u, ret=True)
        x_recons = self.linear_head(tokens[:, 1:, :])
        out_sum = torch.zeros(x_recons.shape[0], 128, 81924, dtype=x_recons.dtype, device=x_recons.device)
        x_recons = x_recons.reshape(x_recons.shape[0], 640, 128, 153).permute(0, 2, 1, 3).reshape(
            [x_recons.shape[0], 128, -1])
        self.parcel_index = self.parcel_index.to(x_recons.device)
        out_sum.scatter_add_(dim=2,
                             index=self.parcel_index.unsqueeze(0).unsqueeze(0).expand(x_recons.shape[0], 128, -1),
                             src=x_recons)
        counts = torch.bincount(self.parcel_index, minlength=81924)
        out = out_sum / counts
        out = self.predictor(out)
        return {
            "img_embed_u": x,
            "cls": out
        }

class Finetuning_parcellation_ACLIP(nn.Module):
    def __init__(self, hidden_dim, visual, output_channel=2):
        super(Finetuning_parcellation_ACLIP, self).__init__()
        self.visual = visual

        # self.linear_head = nn.Sequential(
        #     nn.Linear(hidden_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(128, output_channel),
        # )
        # self.initiate_parameters()
        # self.parcel_index = Get_parcellation()
        # self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962], axis=0)
        self.linear_head = nn.Linear(hidden_dim, 128 * 153, bias=True)
        self.predictor = nn.Conv1d(128, 36, kernel_size=1, bias=True)
        self.parcel_index_1 = Get_parcellation()
        self.parcel_index = torch.tensor(
            np.concatenate([self.parcel_index_1, self.parcel_index_1 + 40962], axis=0)).reshape([-1]).long()


    def encode_image(self, image, mask=None, ret=False, ema=False):
        if ema == False:
            x, attn, ids_restore, mask = self.visual(image, mask=mask)
            tokens = x
            x = x[:, 0]
        else:
            x, attn, ids_restore, mask = self.visual_ema(image, mask=mask)
            tokens = x
            x = x[:, 0]

        if ret:
            return x, attn, ids_restore, tokens, mask
        return x


    def initiate_parameters(self):
        for name, module in self.linear_head.named_children():
            # 只对Linear层进行初始化
            if isinstance(module, nn.Linear):
                # 初始化权重
                nn.init.xavier_uniform_(module.weight)
                # 初始化偏置 (统一设置为0.1)
                nn.init.constant_(module.bias, 0.1)


    def forward(self, u):
        x_parcel = u[:, :, self.parcel_index]
        u = x_parcel.transpose(-1, -2).contiguous().reshape([x_parcel.shape[0], -1, 640])
        x, attn, ids_restore, tokens, mask = self.encode_image(u, ret=True)
        x_recons = self.linear_head(tokens[:, 1:, :])
        out_sum = torch.zeros(x_recons.shape[0], 128, 81924, dtype=x_recons.dtype, device=x_recons.device)
        x_recons = x_recons.reshape(x_recons.shape[0], 640, 128, 153).permute(0, 2, 1, 3).reshape(
            [x_recons.shape[0], 128, -1])
        self.parcel_index = self.parcel_index.to(x_recons.device)
        out_sum.scatter_add_(dim=2,
                             index=self.parcel_index.unsqueeze(0).unsqueeze(0).expand(x_recons.shape[0], 128, -1),
                             src=x_recons)
        counts = torch.bincount(self.parcel_index, minlength=81924)
        out = out_sum / counts
        out = self.predictor(out)
        return {
            "img_embed_u": x,
            "cls": out
        }


def forward_attn(self, x):

    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, attn.detach()


def forward_block(self, x):
    attn_x, attn = forward_attn(self.attn, self.norm1(x))
    x = x + self.drop_path1(attn_x)
    x = x + self.drop_path2(self.mlp(self.norm2(x)))

    return x, attn

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,in_chans, embed_dim=512, norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_patches = 640
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=self.num_patches)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        self.num_patches = x.shape[1]
        x = self.proj(x)
        x = self.norm(x.transpose(1, 2)).contiguous()
        return x

class MaskVisionTransformer(VisionTransformer):
    def __init__(self,pos_embed, mask_ratio=0.0, input_channels=6, **kwargs):
        super(MaskVisionTransformer, self).__init__(**kwargs)
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbed(input_channels, embed_dim=kwargs.get("embed_dim", 512), norm_layer=nn.LayerNorm)
        num_patches = self.patch_embed.num_patches
        embed_len = num_patches if self.no_embed_class else num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, self.embed_dim) * .02)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def mask_model(self, x, mask):
        N, L, D = x.shape  # batch, length, dim
        ids = torch.argsort(mask.long(), dim=1)  # ascend
        ids_restore = torch.argsort(ids, dim=1)
        mask_len = mask[0].sum()
        ids_keep = ids[:, : L - mask_len]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked, ids_restore

    def forward_features(self, x, mask=None, need_attn=False):
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        ids_restore = None

        # only student masks and even that only during training
        if self.mask_ratio > 0 and self.training is True:
            if mask is None:
                x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
            else:
                x, ids_restore = self.mask_model(x, mask)

        # add pos embed and cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        attn_list = []
        if need_attn:
            for b in self.blocks:
                x, attn_now = forward_block(b, x)
                attn_list.append(attn_now)
            attn = torch.stack(attn_list, dim=0)
            attn = torch.mean(attn, dim=0)
            attn = attn[:, :, 0, 1:].mean(1).detach().clone()
            x = self.norm(x)
            return x, attn, ids_restore, mask
        else:
            x = self.blocks(x)
            x = self.norm(x)
            attn = None
            return x, attn, ids_restore, mask

    def forward(self, x, mask=None, need_attn=False):
        x = self.forward_features(x, mask=mask, need_attn=need_attn)
        # x = self.head(x)
        return x

@register_model
def mask_vit_small_patch16_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch16_224",
        MaskVisionTransformer,
        pretrained=pretrained,
        **model_kwargs,
    )
    return model

@register_model
def mask_vit_base_patch16_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224",
        MaskVisionTransformer,
        pretrained=pretrained,
        **model_kwargs,
    )
    return model

@register_model
def mask_vit_large_patch16_224(pretrained=False, **kwargs):
    """ViT-Large (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_224",
        MaskVisionTransformer,
        pretrained=pretrained,
        **model_kwargs,
    )
    return model


@register_model
def mask_vit_base_patch32_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_224",
        MaskVisionTransformer,
        pretrained=pretrained,
        **model_kwargs,
    )
    return model

def _create_vision_transformer(
    variant,
    transformer=MaskVisionTransformer,
    pretrained=False,
    default_cfg=None,
    **kwargs,
):
    default_cfg = default_cfg or default_cfgs[variant] # this will hold some config info about model, ex. ViT-S's input size, interpolation, and url to get it.
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )
    print(default_cfg)
    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg["num_classes"]
    num_classes = kwargs.get("num_classes", default_num_classes)
    repr_size = kwargs.pop("representation_size", None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        print("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        transformer,
        variant,
        pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in default_cfg["url"],
        **kwargs,
    )
    return model


def ACLIP_VITB16(mask_ratio=0, **kwargs):
    vision_model = MaskVisionTransformer(pos_embed="learn", mask_ratio=mask_ratio, embed_dim=768, depth=12, num_heads=12, input_channels=153)
    vision_model_ema = MaskVisionTransformer(pos_embed="learn", mask_ratio=0, embed_dim=768, depth=12, num_heads=12, input_channels=153)
    model = ACLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        vision_model_ema=vision_model_ema,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        mask_ratio=mask_ratio,
        **kwargs,
    )
    return model

def Finetuning_ACLIP_VITB16():
    vision_model = MaskVisionTransformer(pos_embed="learn", embed_dim=768, depth=12, num_heads=12, input_channels=153)
    # vision_model_ema = MaskVisionTransformer(pos_embed="learn", mask_ratio=0, embed_dim=768, depth=12, num_heads=12)
    model = Finetuning_ACLIP(
        hidden_dim=768,
        visual = vision_model,
        output_channel = 2,
    )
    return model

def Finetuning_Regression_ACLIP_VITB16():
    vision_model = MaskVisionTransformer(pos_embed="learn", embed_dim=768, depth=12, num_heads=12,input_channels=153)
    # vision_model_ema = MaskVisionTransformer(pos_embed="learn", mask_ratio=0, embed_dim=768, depth=12, num_heads=12)
    model = Finetuning_ACLIP(
        hidden_dim=768,
        visual = vision_model,
        output_channel = 1,
    )
    return model

def finetune_parcellation_ACLIP(mask_ratio=0, **kwargs):
    vision_model = MaskVisionTransformer(pos_embed="learn", mask_ratio=mask_ratio, embed_dim=768, depth=12,
                                         num_heads=12, input_channels=153)
    model = Finetuning_parcellation_ACLIP(hidden_dim=768,visual=vision_model)
    return model

def CLIP_VITB16(**kwargs):
    vision_model = MaskVisionTransformer(pos_embed="learn", mask_ratio=0, embed_dim=768, depth=12,
                                         num_heads=12,input_channels=153)
    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )
    return model


def Finetuning_CLIP_VITB16(**kwargs):
    vision_model = MaskVisionTransformer(pos_embed="learn", mask_ratio=0, embed_dim=768, depth=12,
                                         num_heads=12,input_channels=153)
    model = Finetuning_CLIP(
        768, vision_model, output_channel=2
    )
    return model

def Finetuning_parcellation_CLIP_VITB16(**kwargs):
    vision_model = MaskVisionTransformer(pos_embed="learn", mask_ratio=0, embed_dim=768, depth=12,
                                         num_heads=12,input_channels=153)
    model = Finetuning_parcellation_CLIP(
        768, vision_model
    )
    return model

def Finetuning_Regression_CLIP_VITB16(**kwargs):
    vision_model = MaskVisionTransformer(pos_embed="learn", mask_ratio=0, embed_dim=768, depth=12,
                                         num_heads=12,input_channels=153)
    model = Finetuning_CLIP(
        768, vision_model, output_channel=1
    )
    return model


def ACLIP_VITL16(mask_ratio=0, **kwargs):
    vision_model = timm.create_model(
        "mask_vit_large_patch16_224", num_classes=0, mask_ratio=mask_ratio
    )
    vision_model_ema = timm.create_model(
        'mask_vit_large_patch16_224', num_classes=0, mask_ratio=0
    )
    model = ACLIP(
        embed_dim=512,
        vision_width=1024,
        vision_model=vision_model,
        vision_model_ema=vision_model_ema,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        mask_ratio=mask_ratio,
        **kwargs,
    )
    return model

# SMALL
def ACLIP_VITS16(mask_ratio=0, **kwargs):
    # num_classes is set to 0 since we are doing self-supervised learning. No need for classification head.
    vision_model = timm.create_model(
        "mask_vit_small_patch16_224", num_classes=0, mask_ratio=mask_ratio
    )
    vision_model_ema = timm.create_model(
        "mask_vit_small_patch16_224", num_classes=0, mask_ratio=0
    )
    model = ACLIP(
        embed_dim=512,
        vision_width=384,
        vision_model=vision_model,
        vision_model_ema=vision_model_ema,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        mask_ratio=mask_ratio,
        **kwargs,
    )
    return model


def DetailCLIP_VITB16(mask_ratio=0, **kwargs):
    # vision_model = timm.create_model(
    #     "mask_vit_base_patch16_224", num_classes=0, mask_ratio=mask_ratio
    # )
    # vision_model_ema = timm.create_model(
    #     'mask_vit_base_patch16_224', num_classes=0, mask_ratio=0
    # )
    vision_model = MaskVisionTransformer(pos_embed="learn", mask_ratio=mask_ratio, embed_dim=768, depth=12, num_heads=12,input_channels=153)
    vision_model_ema = MaskVisionTransformer(pos_embed="learn", mask_ratio=0, embed_dim=768, depth=12, num_heads=12,input_channels=153)
    model = DetailCLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        vision_model_ema=vision_model_ema,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        mask_ratio=mask_ratio,
        # decoder params
        decoder_depth=1, 
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # for reconstruction projection head
        patch_size = 16,
        in_chans = 1,
        # for iBOT projection head
        out_dim = 8192,
        patch_out_dim = 8192,
        norm_in_head = None,
        act_in_head = 'gelu',
        shared_head_teacher = True,
        **kwargs)
    return model

def finetune_DetailCLIP(mask_ratio=0, **kwargs):
    vision_model = MaskVisionTransformer(pos_embed="learn", mask_ratio=mask_ratio, embed_dim=768, depth=12,
                                         num_heads=12, input_channels=153)
    model = Finetuning_DetailCLIP(hidden_dim=768,visual=vision_model,output_channel=2)
    return model

def finetune_regression_DetailCLIP(mask_ratio=0, **kwargs):
    vision_model = MaskVisionTransformer(pos_embed="learn", mask_ratio=mask_ratio, embed_dim=768, depth=12,
                                         num_heads=12, input_channels=153)
    model = Finetuning_DetailCLIP(hidden_dim=768,visual=vision_model,output_channel=1)
    return model

def finetune_parcellation_DetailCLIP(mask_ratio=0, **kwargs):
    vision_model = MaskVisionTransformer(pos_embed="learn", mask_ratio=mask_ratio, embed_dim=768, depth=12,
                                         num_heads=12, input_channels=153)
    model = Finetuning_parcellation_DetailCLIP(hidden_dim=768,visual=vision_model)
    return model
