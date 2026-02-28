import time
from functools import partial

from torch import nn
from collections import OrderedDict
import torch
import numpy as np
from timm.models.vision_transformer import VisionTransformer, Block
import os
from TokenGenerator import SurfEncoder, TokenGeneratorUnit, TokenGeneratorUnit_single, \
    TokenGeneratorUnit_single_woSpherical, TokenGeneratorUnit_single_woTransformer, TokenGeneratorUnit_single_woScalar
from util.utils import Get_parcellation
from utils import iBOTHead, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

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
        global_mean_path,
        global_std_path,
        # embed_dim: int,
        # vision
        # vision_width: int,
        # vision_model: nn.Module,
        # text
        # context_length: int,
        # vocab_size: int,
        # transformer_width: int,
        # transformer_heads: int,
        # transformer_layers: int,
        Pretrain = True,
    ):
        super().__init__()

        # self.context_length = context_length
        # self.vision_width = vision_width
        mean = torch.tensor(np.load(global_mean_path).reshape([1,6,1]).astype(np.float32))
        std = torch.tensor(np.load(global_std_path).reshape([1,6,1]).astype(np.float32))
        self.spherical_encoder = SurfEncoder(6, 512, 320, None, 32,1.1, mean,std,
                               2,4,128,256,0.1, 4,6,512,128,0.1,Pretrain)
        # self.clip,_ = clip.load("ViT-B/32")
        # for p in self.clip.parameters():
        #     p.requires_grad = False


        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # self.initialize_parameters()
    #
    # def initialize_parameters(self):
        # nn.init.normal_(self.token_embedding.weight, std=0.02) # each weight is independently and randomly sampled from normal dist of specified std
        # nn.init.normal_(self.positional_embedding, std=0.01)

        # proj_std = (self.transformer.width**-0.5) * (
        #     (2 * self.transformer.layers) ** -0.5
        # ) # the product of 1/sqrt(transformer width) and 1/sqrt(2* number of transformer layers)
        # attn_std = self.transformer.width**-0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std) # in_proj couples Q,K, and V. So, we initialize them with the same std.
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std) # out_proj is the linear
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # nn.init.normal_(self.image_projection, std=self.vision_width**-0.5)
        # nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image):
        x = self.spherical_encoder(image)
        return x

    def encode_text(self, text):
        # text_tokens = clip.tokenize(text)
        with torch.no_grad():
            # text_embedding = self.clip.encode_text(text)
            return text

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "logit_scale": self.logit_scale.exp(),
        }

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
    ):
        super().__init__()
        self.num_patches = 640

    def forward(self, x):
        self.num_patches = x.shape[1]
        return x

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

class PatchEmbed_for_ablation(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,input_channels,output_channels,
    ):
        super().__init__()
        self.num_patches = 640
        self.linear = nn.Conv1d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        self.num_patches = x.shape[1]
        x = self.linear(x)
        return x.transpose(1, 2).contiguous()

class MaskVisionTransformer_for_ablation(VisionTransformer):
    def __init__(self, pos_embed, mask_ratio=0, **kwargs): # embed_dim, depth, number_heads
        super(MaskVisionTransformer_for_ablation, self).__init__(**kwargs)
        self.mask_ratio = mask_ratio
        for param in self.patch_embed.proj.parameters():
            param.requires_grad = False
        self.patch_embed = PatchEmbed_for_ablation(1*153,512)
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
        if self.mask_ratio > 0:
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
        if need_attn: # the teacher should be this
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

class MaskVisionTransformer(VisionTransformer):
    def __init__(self, pos_embed, mask_ratio=0, **kwargs): # embed_dim, depth, number_heads
        super(MaskVisionTransformer, self).__init__(**kwargs)
        self.mask_ratio = mask_ratio
        for param in self.patch_embed.proj.parameters():
            param.requires_grad = False
        self.patch_embed = PatchEmbed()
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
        if self.mask_ratio > 0:
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
        if need_attn: # the teacher should be this
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


def get_att_mask(attention, ratio=0.5):
    # attention size = (B, N)
    bs = attention.shape[0]
    masks = torch.ones((attention.shape), dtype=torch.bool, device=attention.device)
    N = int(attention.shape[1] * ratio)
    reservation = torch.argsort(attention, descending=True)
    reservation = reservation[:, :N + 1]  # get top N values
    masks = masks.scatter_(1, reservation, False)
    return masks



class SurfaceCLIP(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            # Decoder params
            decoder_depth,
            decoder_num_heads,
            # proj head
            out_dim,
            norm_in_head,
            act_in_head,
            shared_head_teacher,
            nlayers,
            proj_hidden_dim,
            **kwargs,
    ):
        super().__init__()
        self.mask_ratio = kwargs["mask_ratio"]
        self.tokenizer = TokenGeneratorUnit(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                            TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                            global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                            TU_dim_head, TU_mlp_dim, TU_dropout)

        self.visual_ema = MaskVisionTransformer(pos_embed="learn", mask_ratio=0, embed_dim=hidden_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.visual = MaskVisionTransformer(pos_embed="learn", mask_ratio=self.mask_ratio, embed_dim=hidden_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.image_projection = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.image_projection_e = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.hidden_dim = hidden_dim
        self.image_projection_e.requires_grad = False
        self.image_projection_e.data.copy_(self.image_projection.data)
        for param_m, param_b in zip(self.visual_ema.parameters(), self.visual.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        self.initialize_parameters()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        # --------------------------------------------------------------------------
        # decoder specifics
        mlp_ratio = 4


        self.decoder_embed = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        num_patches = self.visual.patch_embed.num_patches

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(hidden_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm,eps=1e-6))
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(hidden_dim,eps=1e-6)
        self.initialize_decoder()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # projection heads
        # MAE projection head
        print('\tCreating MAE projection head')
        self.reconstruction_pred = nn.Linear(hidden_dim, input_channel*153, bias=True)  # MSE loss
        print('\tMAE projection head created')

        # IBOT projection head
        print('\tCreating IBOT projection head')
        self.ibot_head = iBOTHead(
            hidden_dim,
            out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
            nlayers=nlayers,
            hidden_dim=proj_hidden_dim
        )
        self.ibot_head_e = iBOTHead(
            hidden_dim,
            out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
            nlayers=nlayers,
            hidden_dim=proj_hidden_dim
        )
        result = self.ibot_head_e.load_state_dict(self.ibot_head.state_dict(), strict=False)
        print('\tkeys have been loaded for ibot head with status:', result)
        print('\tIBOT projection head created')
        for p in self.ibot_head_e.parameters():
            p.requires_grad = False
        # --------------------------------------------------------------------------
        print('\tDetailCLIP model created')

    def initialize_parameters(self):
        nn.init.normal_(self.visual.pos_embed, std=0.01)
        nn.init.normal_(self.image_projection, std=self.hidden_dim**-0.5)

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
                self.ibot_head.parameters(), self.ibot_head_e.parameters()
        ):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)

        self.image_projection_e.data = self.image_projection_e.data * m + self.image_projection * (1.0 - m)

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
        with torch.no_grad():
            return text

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

    def forward(self, u, text, momentum):
        # u are two augmented images
        surf = u
        u = self.tokenizer(u)
        with torch.no_grad():
            self._update_momentum_encoder(momentum)
            u_e, attn_u, _, _ = self.visual_ema(u, need_attn=True)
            teacher_ibot = self.ibot_head_e(u_e)

        # obtain masks
        mask_u = get_att_mask(attn_u, ratio=self.mask_ratio)
        img_embed_u, _, ids_restore_u, latent_u, _ = self.encode_image(u, mask=mask_u, ret=True)
        # print(ids_restore_u,latent_u)
        u_s = self.forward_decoder(latent_u, ids_restore_u)
        u_s_reconstructed = self.reconstruction_pred(u_s)[:, 1:, :]

        u_s[:, 0] = latent_u[:,0]  # assigning the cls token of u_s to latent_u's cls token to keep the cls token same for ibot task
        student_ibot = self.ibot_head(u_s)

        text_embed = self.encode_text(text)

        return {
            # images
            "u": surf,

            # CLIP outputs
            "img_embed_u": img_embed_u,
            "text_embed": text_embed,
            "logit_scale": self.logit_scale.exp(),

            # IBOT outputs
            "teacher_ibot": teacher_ibot,
            "student_ibot": student_ibot,

            # MAE outputs
            "u_s_reconstructed": u_s_reconstructed,

            # masks
            "mask_u": mask_u,
        }

class SurfaceCLIP_single(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            # Decoder params
            decoder_depth,
            decoder_num_heads,
            # proj head
            out_dim,
            norm_in_head,
            act_in_head,
            shared_head_teacher,
            nlayers,
            proj_hidden_dim,
            **kwargs,
    ):
        super().__init__()
        self.mask_ratio = kwargs["mask_ratio"]
        self.tokenizer = TokenGeneratorUnit_single(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                            TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                            global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                            TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)

        self.visual_ema = MaskVisionTransformer(pos_embed="learn", mask_ratio=0, embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.visual = MaskVisionTransformer(pos_embed="learn", mask_ratio=self.mask_ratio, embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.image_projection = nn.Parameter(torch.empty(TU_out_dim, TU_out_dim))
        self.image_projection_e = nn.Parameter(torch.empty(TU_out_dim, TU_out_dim))
        self.hidden_dim = hidden_dim
        self.image_projection_e.requires_grad = False
        self.image_projection_e.data.copy_(self.image_projection.data)
        for param_m, param_b in zip(self.visual_ema.parameters(), self.visual.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        self.initialize_parameters()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        # --------------------------------------------------------------------------
        # decoder specifics
        mlp_ratio = 4


        self.decoder_embed = nn.Linear(TU_out_dim, TU_out_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, TU_out_dim))
        num_patches = self.visual.patch_embed.num_patches

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, TU_out_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(TU_out_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm,eps=1e-6))
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(TU_out_dim,eps=1e-6)
        self.initialize_decoder()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # projection heads
        # MAE projection head
        print('\tCreating MAE projection head')
        self.reconstruction_pred = nn.Linear(TU_out_dim, input_channel*153, bias=True)  # MSE loss
        print('\tMAE projection head created')

        # IBOT projection head
        print('\tCreating IBOT projection head')
        self.ibot_head = iBOTHead(
            TU_out_dim,
            out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
            nlayers=nlayers,
            hidden_dim=proj_hidden_dim
        )
        self.ibot_head_e = iBOTHead(
            TU_out_dim,
            out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
            nlayers=nlayers,
            hidden_dim=proj_hidden_dim
        )
        result = self.ibot_head_e.load_state_dict(self.ibot_head.state_dict(), strict=False)
        print('\tkeys have been loaded for ibot head with status:', result)
        print('\tIBOT projection head created')
        for p in self.ibot_head_e.parameters():
            p.requires_grad = False
        # --------------------------------------------------------------------------
        print('\tDetailCLIP model created')

    def initialize_parameters(self):
        nn.init.normal_(self.visual.pos_embed, std=0.01)
        nn.init.normal_(self.image_projection, std=self.hidden_dim**-0.5)

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
                self.ibot_head.parameters(), self.ibot_head_e.parameters()
        ):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)

        self.image_projection_e.data = self.image_projection_e.data * m + self.image_projection * (1.0 - m)

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
        with torch.no_grad():
            return text

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

    def forward(self, u, text, momentum):
        surf = u
        u = self.tokenizer(u)
        with torch.no_grad():
            self._update_momentum_encoder(momentum)
            u_e, attn_u, _, _ = self.visual_ema(u, need_attn=True)
            teacher_ibot = self.ibot_head_e(u_e)

        # obtain masks
        mask_u = get_att_mask(attn_u, ratio=self.mask_ratio)
        img_embed_u, _, ids_restore_u, latent_u, _ = self.encode_image(u, mask=mask_u, ret=True)
        # print(ids_restore_u,latent_u)
        u_s = self.forward_decoder(latent_u, ids_restore_u)
        u_s_reconstructed = self.reconstruction_pred(u_s)[:, 1:, :]

        u_s[:, 0] = latent_u[:,0]  # assigning the cls token of u_s to latent_u's cls token to keep the cls token same for ibot task
        student_ibot = self.ibot_head(u_s)

        text_embed = self.encode_text(text)

        return {
            # images
            "u": surf,

            # CLIP outputs
            "img_embed_u": img_embed_u,
            "text_embed": text_embed,
            "logit_scale": self.logit_scale.exp(),

            # IBOT outputs
            "teacher_ibot": teacher_ibot,
            "student_ibot": student_ibot,

            # MAE outputs
            "u_s_reconstructed": u_s_reconstructed,

            # masks
            "mask_u": mask_u,
        }

class SurfaceCLIP_single_woTokenizer(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            # Decoder params
            decoder_depth,
            decoder_num_heads,
            # proj head
            out_dim,
            norm_in_head,
            act_in_head,
            shared_head_teacher,
            nlayers,
            proj_hidden_dim,
            **kwargs,
    ):
        super().__init__()
        self.mask_ratio = kwargs["mask_ratio"]
        # self.tokenizer = TokenGeneratorUnit_single(input_channel, hidden_dim, num_patches, TU_scalar_scales,
        #                                     TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
        #                                     global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
        #                                     TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)

        self.visual_ema = MaskVisionTransformer_for_ablation(pos_embed="learn", mask_ratio=0, embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.visual = MaskVisionTransformer_for_ablation(pos_embed="learn", mask_ratio=self.mask_ratio, embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.image_projection = nn.Parameter(torch.empty(TU_out_dim, TU_out_dim))
        self.image_projection_e = nn.Parameter(torch.empty(TU_out_dim, TU_out_dim))
        self.hidden_dim = hidden_dim
        self.image_projection_e.requires_grad = False
        self.image_projection_e.data.copy_(self.image_projection.data)
        for param_m, param_b in zip(self.visual_ema.parameters(), self.visual.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        self.initialize_parameters()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.parcel_index = Get_parcellation()
        # self.neighbor = np.concatenate([self.neighbor, self.neighbor + 40962])
        self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962])

        # --------------------------------------------------------------------------
        # decoder specifics
        mlp_ratio = 4


        self.decoder_embed = nn.Linear(TU_out_dim, TU_out_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, TU_out_dim))
        num_patches = self.visual.patch_embed.num_patches

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, TU_out_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(TU_out_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm,eps=1e-6))
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(TU_out_dim,eps=1e-6)
        self.initialize_decoder()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # projection heads
        # MAE projection head
        print('\tCreating MAE projection head')
        self.reconstruction_pred = nn.Linear(TU_out_dim, input_channel*153, bias=True)  # MSE loss
        print('\tMAE projection head created')

        # IBOT projection head
        print('\tCreating IBOT projection head')
        self.ibot_head = iBOTHead(
            TU_out_dim,
            out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
            nlayers=nlayers,
            hidden_dim=proj_hidden_dim
        )
        self.ibot_head_e = iBOTHead(
            TU_out_dim,
            out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
            nlayers=nlayers,
            hidden_dim=proj_hidden_dim
        )
        result = self.ibot_head_e.load_state_dict(self.ibot_head.state_dict(), strict=False)
        print('\tkeys have been loaded for ibot head with status:', result)
        print('\tIBOT projection head created')
        for p in self.ibot_head_e.parameters():
            p.requires_grad = False
        # --------------------------------------------------------------------------
        print('\tDetailCLIP model created')

    def initialize_parameters(self):
        nn.init.normal_(self.visual.pos_embed, std=0.01)
        nn.init.normal_(self.image_projection, std=self.hidden_dim**-0.5)

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
                self.ibot_head.parameters(), self.ibot_head_e.parameters()
        ):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)

        self.image_projection_e.data = self.image_projection_e.data * m + self.image_projection * (1.0 - m)

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
        with torch.no_grad():
            return text

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

    def forward(self, u, text, momentum):
        surf = u
        # u = self.tokenizer(u)
        u = u[:,:,self.parcel_index].transpose(-1,-2).reshape([surf.shape[0], -1, 640])
        with torch.no_grad():
            self._update_momentum_encoder(momentum)
            u_e, attn_u, _, _ = self.visual_ema(u, need_attn=True)
            teacher_ibot = self.ibot_head_e(u_e)

        # obtain masks
        mask_u = get_att_mask(attn_u, ratio=self.mask_ratio)
        img_embed_u, _, ids_restore_u, latent_u, _ = self.encode_image(u, mask=mask_u, ret=True)
        # print(ids_restore_u,latent_u)
        u_s = self.forward_decoder(latent_u, ids_restore_u)
        u_s_reconstructed = self.reconstruction_pred(u_s)[:, 1:, :]

        u_s[:, 0] = latent_u[:,0]  # assigning the cls token of u_s to latent_u's cls token to keep the cls token same for ibot task
        student_ibot = self.ibot_head(u_s)

        text_embed = self.encode_text(text)

        return {
            # images
            "u": surf,

            # CLIP outputs
            "img_embed_u": img_embed_u,
            "text_embed": text_embed,
            "logit_scale": self.logit_scale.exp(),

            # IBOT outputs
            "teacher_ibot": teacher_ibot,
            "student_ibot": student_ibot,

            # MAE outputs
            "u_s_reconstructed": u_s_reconstructed,

            # masks
            "mask_u": mask_u,
        }

class SurfaceCLIP_single_woSpherical(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            # Decoder params
            decoder_depth,
            decoder_num_heads,
            # proj head
            out_dim,
            norm_in_head,
            act_in_head,
            shared_head_teacher,
            nlayers,
            proj_hidden_dim,
            **kwargs,
    ):
        super().__init__()
        self.mask_ratio = kwargs["mask_ratio"]
        self.tokenizer = TokenGeneratorUnit_single_woSpherical(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                            TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                            global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                            TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)

        self.visual_ema = MaskVisionTransformer(pos_embed="learn", mask_ratio=0, embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.visual = MaskVisionTransformer(pos_embed="learn", mask_ratio=self.mask_ratio, embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.image_projection = nn.Parameter(torch.empty(TU_out_dim, TU_out_dim))
        self.image_projection_e = nn.Parameter(torch.empty(TU_out_dim, TU_out_dim))
        self.hidden_dim = hidden_dim
        self.image_projection_e.requires_grad = False
        self.image_projection_e.data.copy_(self.image_projection.data)
        for param_m, param_b in zip(self.visual_ema.parameters(), self.visual.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        self.initialize_parameters()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        # --------------------------------------------------------------------------
        # decoder specifics
        mlp_ratio = 4


        self.decoder_embed = nn.Linear(TU_out_dim, TU_out_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, TU_out_dim))
        num_patches = self.visual.patch_embed.num_patches

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, TU_out_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(TU_out_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm,eps=1e-6))
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(TU_out_dim,eps=1e-6)
        self.initialize_decoder()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # projection heads
        # MAE projection head
        print('\tCreating MAE projection head')
        self.reconstruction_pred = nn.Linear(TU_out_dim, input_channel*153, bias=True)  # MSE loss
        print('\tMAE projection head created')

        # IBOT projection head
        print('\tCreating IBOT projection head')
        self.ibot_head = iBOTHead(
            TU_out_dim,
            out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
            nlayers=nlayers,
            hidden_dim=proj_hidden_dim
        )
        self.ibot_head_e = iBOTHead(
            TU_out_dim,
            out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
            nlayers=nlayers,
            hidden_dim=proj_hidden_dim
        )
        result = self.ibot_head_e.load_state_dict(self.ibot_head.state_dict(), strict=False)
        print('\tkeys have been loaded for ibot head with status:', result)
        print('\tIBOT projection head created')
        for p in self.ibot_head_e.parameters():
            p.requires_grad = False
        # --------------------------------------------------------------------------
        print('\tDetailCLIP model created')

    def initialize_parameters(self):
        nn.init.normal_(self.visual.pos_embed, std=0.01)
        nn.init.normal_(self.image_projection, std=self.hidden_dim**-0.5)

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
                self.ibot_head.parameters(), self.ibot_head_e.parameters()
        ):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)

        self.image_projection_e.data = self.image_projection_e.data * m + self.image_projection * (1.0 - m)

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
        with torch.no_grad():
            return text

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

    def forward(self, u, text, momentum):
        surf = u
        u = self.tokenizer(u)
        with torch.no_grad():
            self._update_momentum_encoder(momentum)
            u_e, attn_u, _, _ = self.visual_ema(u, need_attn=True)
            teacher_ibot = self.ibot_head_e(u_e)

        # obtain masks
        mask_u = get_att_mask(attn_u, ratio=self.mask_ratio)
        img_embed_u, _, ids_restore_u, latent_u, _ = self.encode_image(u, mask=mask_u, ret=True)
        # print(ids_restore_u,latent_u)
        u_s = self.forward_decoder(latent_u, ids_restore_u)
        u_s_reconstructed = self.reconstruction_pred(u_s)[:, 1:, :]

        u_s[:, 0] = latent_u[:,0]  # assigning the cls token of u_s to latent_u's cls token to keep the cls token same for ibot task
        student_ibot = self.ibot_head(u_s)

        text_embed = self.encode_text(text)

        return {
            # images
            "u": surf,

            # CLIP outputs
            "img_embed_u": img_embed_u,
            "text_embed": text_embed,
            "logit_scale": self.logit_scale.exp(),

            # IBOT outputs
            "teacher_ibot": teacher_ibot,
            "student_ibot": student_ibot,

            # MAE outputs
            "u_s_reconstructed": u_s_reconstructed,

            # masks
            "mask_u": mask_u,
        }

class SurfaceCLIP_single_woTransformer(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            # Decoder params
            decoder_depth,
            decoder_num_heads,
            # proj head
            out_dim,
            norm_in_head,
            act_in_head,
            shared_head_teacher,
            nlayers,
            proj_hidden_dim,
            **kwargs,
    ):
        super().__init__()
        self.mask_ratio = kwargs["mask_ratio"]
        self.tokenizer = TokenGeneratorUnit_single_woTransformer(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                            TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                            global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                            TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)

        self.visual_ema = MaskVisionTransformer(pos_embed="learn", mask_ratio=0, embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.visual = MaskVisionTransformer(pos_embed="learn", mask_ratio=self.mask_ratio, embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.image_projection = nn.Parameter(torch.empty(TU_out_dim, TU_out_dim))
        self.image_projection_e = nn.Parameter(torch.empty(TU_out_dim, TU_out_dim))
        self.hidden_dim = hidden_dim
        self.image_projection_e.requires_grad = False
        self.image_projection_e.data.copy_(self.image_projection.data)
        for param_m, param_b in zip(self.visual_ema.parameters(), self.visual.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        self.initialize_parameters()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        # --------------------------------------------------------------------------
        # decoder specifics
        mlp_ratio = 4


        self.decoder_embed = nn.Linear(TU_out_dim, TU_out_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, TU_out_dim))
        num_patches = self.visual.patch_embed.num_patches

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, TU_out_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(TU_out_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm,eps=1e-6))
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(TU_out_dim,eps=1e-6)
        self.initialize_decoder()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # projection heads
        # MAE projection head
        print('\tCreating MAE projection head')
        self.reconstruction_pred = nn.Linear(TU_out_dim, input_channel*153, bias=True)  # MSE loss
        print('\tMAE projection head created')

        # IBOT projection head
        print('\tCreating IBOT projection head')
        self.ibot_head = iBOTHead(
            TU_out_dim,
            out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
            nlayers=nlayers,
            hidden_dim=proj_hidden_dim
        )
        self.ibot_head_e = iBOTHead(
            TU_out_dim,
            out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
            nlayers=nlayers,
            hidden_dim=proj_hidden_dim
        )
        result = self.ibot_head_e.load_state_dict(self.ibot_head.state_dict(), strict=False)
        print('\tkeys have been loaded for ibot head with status:', result)
        print('\tIBOT projection head created')
        for p in self.ibot_head_e.parameters():
            p.requires_grad = False
        # --------------------------------------------------------------------------
        print('\tDetailCLIP model created')

    def initialize_parameters(self):
        nn.init.normal_(self.visual.pos_embed, std=0.01)
        nn.init.normal_(self.image_projection, std=self.hidden_dim**-0.5)

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
                self.ibot_head.parameters(), self.ibot_head_e.parameters()
        ):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)

        self.image_projection_e.data = self.image_projection_e.data * m + self.image_projection * (1.0 - m)

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
        with torch.no_grad():
            return text

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

    def forward(self, u, text, momentum):
        surf = u
        u = self.tokenizer(u)
        with torch.no_grad():
            self._update_momentum_encoder(momentum)
            u_e, attn_u, _, _ = self.visual_ema(u, need_attn=True)
            teacher_ibot = self.ibot_head_e(u_e)

        # obtain masks
        mask_u = get_att_mask(attn_u, ratio=self.mask_ratio)
        img_embed_u, _, ids_restore_u, latent_u, _ = self.encode_image(u, mask=mask_u, ret=True)
        # print(ids_restore_u,latent_u)
        u_s = self.forward_decoder(latent_u, ids_restore_u)
        u_s_reconstructed = self.reconstruction_pred(u_s)[:, 1:, :]

        u_s[:, 0] = latent_u[:,0]  # assigning the cls token of u_s to latent_u's cls token to keep the cls token same for ibot task
        student_ibot = self.ibot_head(u_s)

        text_embed = self.encode_text(text)

        return {
            # images
            "u": surf,

            # CLIP outputs
            "img_embed_u": img_embed_u,
            "text_embed": text_embed,
            "logit_scale": self.logit_scale.exp(),

            # IBOT outputs
            "teacher_ibot": teacher_ibot,
            "student_ibot": student_ibot,

            # MAE outputs
            "u_s_reconstructed": u_s_reconstructed,

            # masks
            "mask_u": mask_u,
        }

class SurfaceCLIP_single_woScalar(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            # Decoder params
            decoder_depth,
            decoder_num_heads,
            # proj head
            out_dim,
            norm_in_head,
            act_in_head,
            shared_head_teacher,
            nlayers,
            proj_hidden_dim,
            **kwargs,
    ):
        super().__init__()
        self.mask_ratio = kwargs["mask_ratio"]
        self.tokenizer = TokenGeneratorUnit_single_woScalar(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                            TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                            global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                            TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)

        self.visual_ema = MaskVisionTransformer(pos_embed="learn", mask_ratio=0, embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.visual = MaskVisionTransformer(pos_embed="learn", mask_ratio=self.mask_ratio, embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.image_projection = nn.Parameter(torch.empty(TU_out_dim, TU_out_dim))
        self.image_projection_e = nn.Parameter(torch.empty(TU_out_dim, TU_out_dim))
        self.hidden_dim = hidden_dim
        self.image_projection_e.requires_grad = False
        self.image_projection_e.data.copy_(self.image_projection.data)
        for param_m, param_b in zip(self.visual_ema.parameters(), self.visual.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        self.initialize_parameters()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        # --------------------------------------------------------------------------
        # decoder specifics
        mlp_ratio = 4


        self.decoder_embed = nn.Linear(TU_out_dim, TU_out_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, TU_out_dim))
        num_patches = self.visual.patch_embed.num_patches

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, TU_out_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(TU_out_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm,eps=1e-6))
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(TU_out_dim,eps=1e-6)
        self.initialize_decoder()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # projection heads
        # MAE projection head
        print('\tCreating MAE projection head')
        self.reconstruction_pred = nn.Linear(TU_out_dim, input_channel*153, bias=True)  # MSE loss
        print('\tMAE projection head created')

        # IBOT projection head
        print('\tCreating IBOT projection head')
        self.ibot_head = iBOTHead(
            TU_out_dim,
            out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
            nlayers=nlayers,
            hidden_dim=proj_hidden_dim
        )
        self.ibot_head_e = iBOTHead(
            TU_out_dim,
            out_dim,
            norm=norm_in_head,
            act=act_in_head,
            shared_head=shared_head_teacher,
            nlayers=nlayers,
            hidden_dim=proj_hidden_dim
        )
        result = self.ibot_head_e.load_state_dict(self.ibot_head.state_dict(), strict=False)
        print('\tkeys have been loaded for ibot head with status:', result)
        print('\tIBOT projection head created')
        for p in self.ibot_head_e.parameters():
            p.requires_grad = False
        # --------------------------------------------------------------------------
        print('\tDetailCLIP model created')

    def initialize_parameters(self):
        nn.init.normal_(self.visual.pos_embed, std=0.01)
        nn.init.normal_(self.image_projection, std=self.hidden_dim**-0.5)

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
                self.ibot_head.parameters(), self.ibot_head_e.parameters()
        ):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)

        self.image_projection_e.data = self.image_projection_e.data * m + self.image_projection * (1.0 - m)

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
        with torch.no_grad():
            return text

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

    def forward(self, u, text, momentum):
        surf = u
        u = self.tokenizer(u)
        with torch.no_grad():
            self._update_momentum_encoder(momentum)
            u_e, attn_u, _, _ = self.visual_ema(u, need_attn=True)
            teacher_ibot = self.ibot_head_e(u_e)

        # obtain masks
        mask_u = get_att_mask(attn_u, ratio=self.mask_ratio)
        img_embed_u, _, ids_restore_u, latent_u, _ = self.encode_image(u, mask=mask_u, ret=True)
        # print(ids_restore_u,latent_u)
        u_s = self.forward_decoder(latent_u, ids_restore_u)
        u_s_reconstructed = self.reconstruction_pred(u_s)[:, 1:, :]

        u_s[:, 0] = latent_u[:,0]  # assigning the cls token of u_s to latent_u's cls token to keep the cls token same for ibot task
        student_ibot = self.ibot_head(u_s)

        text_embed = self.encode_text(text)

        return {
            # images
            "u": surf,

            # CLIP outputs
            "img_embed_u": img_embed_u,
            "text_embed": text_embed,
            "logit_scale": self.logit_scale.exp(),

            # IBOT outputs
            "teacher_ibot": teacher_ibot,
            "student_ibot": student_ibot,

            # MAE outputs
            "u_s_reconstructed": u_s_reconstructed,

            # masks
            "mask_u": mask_u,
        }

class finetuning_SurfClip_single_channel(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            output_channel=2,
            TU_out_dim=512,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer = TokenGeneratorUnit_single(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                                   TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                                   global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                                   TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)
        self.visual = MaskVisionTransformer(pos_embed="learn", embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.linear_head = nn.Sequential(
            nn.Linear(TU_out_dim, 1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(0.2),
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(0.2),
            # nn.Linear(128, output_channel),
        )
        self.initiate_parameters()

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
        surf = u
        u = self.tokenizer(u)
        img_embed_u = self.encode_image(u)
        cls = self.linear_head(img_embed_u)
        return {
            "u": surf,
            "img_embed_u": img_embed_u,
            "cls": cls
        }

class finetuning_SurfClip_single_channel_woTokenizer(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            output_channel=2,
            TU_out_dim=512,
            **kwargs,
    ):
        super().__init__()
        # self.tokenizer = TokenGeneratorUnit_single(input_channel, hidden_dim, num_patches, TU_scalar_scales,
        #                                            TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
        #                                            global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
        #                                            TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)
        self.visual = MaskVisionTransformer_for_ablation(pos_embed="learn", embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.linear_head = nn.Sequential(
            nn.Linear(TU_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_channel),
        )
        self.initiate_parameters()
        self.parcel_index = Get_parcellation()
        # self.neighbor = np.concatenate([self.neighbor, self.neighbor + 40962])
        self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962])

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
        surf = u
        # u = self.tokenizer(u)
        u = u[:, :, self.parcel_index].transpose(-1, -2).reshape([surf.shape[0], -1, 640])
        img_embed_u = self.encode_image(u)
        cls = self.linear_head(img_embed_u)
        return {
            "u": surf,
            "img_embed_u": img_embed_u,
            "cls": cls
        }

class finetuning_SurfClip_single_channel_woSpherical(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            output_channel=2,
            TU_out_dim=512,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer = TokenGeneratorUnit_single_woSpherical(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                                   TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                                   global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                                   TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)
        self.visual = MaskVisionTransformer(pos_embed="learn", embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.linear_head = nn.Sequential(
            nn.Linear(TU_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_channel),
        )
        self.initiate_parameters()

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
        surf = u
        u = self.tokenizer(u)
        img_embed_u = self.encode_image(u)
        cls = self.linear_head(img_embed_u)
        return {
            "u": surf,
            "img_embed_u": img_embed_u,
            "cls": cls
        }

class finetuning_SurfClip_single_channel_woTransformer(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            output_channel=2,
            TU_out_dim=512,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer = TokenGeneratorUnit_single_woTransformer(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                                   TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                                   global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                                   TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)
        self.visual = MaskVisionTransformer(pos_embed="learn", embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.linear_head = nn.Sequential(
            nn.Linear(TU_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_channel),
        )
        self.initiate_parameters()

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
        surf = u
        u = self.tokenizer(u)
        img_embed_u = self.encode_image(u)
        cls = self.linear_head(img_embed_u)
        return {
            "u": surf,
            "img_embed_u": img_embed_u,
            "cls": cls
        }

class finetuning_SurfClip_single_channel_woScalar(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            output_channel=2,
            TU_out_dim=512,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer = TokenGeneratorUnit_single_woScalar(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                                   TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                                   global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                                   TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)
        self.visual = MaskVisionTransformer(pos_embed="learn", embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.linear_head = nn.Sequential(
            nn.Linear(TU_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_channel),
        )
        self.initiate_parameters()

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
        surf = u
        u = self.tokenizer(u)
        img_embed_u = self.encode_image(u)
        cls = self.linear_head(img_embed_u)
        return {
            "u": surf,
            "img_embed_u": img_embed_u,
            "cls": cls
        }

class finetuning_parcellation_SurfClip_single_channel_woSpherical(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            output_channel=2,
            TU_out_dim=512,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer = TokenGeneratorUnit_single_woSpherical(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                                   TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                                   global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                                   TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)
        self.visual = MaskVisionTransformer(pos_embed="learn", embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.linear_head = nn.Linear(TU_out_dim, 128 * 153, bias=True)
        self.predictor = nn.Conv1d(128, 36, kernel_size=1, bias=True)
        self.parcel_index = Get_parcellation()
        self.parcel_index = torch.tensor(
            np.concatenate([self.parcel_index, self.parcel_index + 40962], axis=0)).reshape([-1]).long()
        self.initiate_parameters()

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
        surf = u
        u = self.tokenizer(u)
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
            "u": surf,
            "img_embed_u": x,
            "cls": out
        }

class finetuning_parcellation_SurfClip_single_channel_woTransformer(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            output_channel=2,
            TU_out_dim=512,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer = TokenGeneratorUnit_single_woTransformer(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                                   TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                                   global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                                   TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)
        self.visual = MaskVisionTransformer(pos_embed="learn", embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.linear_head = nn.Linear(TU_out_dim, 128 * 153, bias=True)
        self.predictor = nn.Conv1d(128, 36, kernel_size=1, bias=True)
        self.parcel_index = Get_parcellation()
        self.parcel_index = torch.tensor(
            np.concatenate([self.parcel_index, self.parcel_index + 40962], axis=0)).reshape([-1]).long()
        self.initiate_parameters()

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
        surf = u
        u = self.tokenizer(u)
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
            "u": surf,
            "img_embed_u": x,
            "cls": out
        }

class finetuning_parcellation_SurfClip_single_channel_woScalar(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            output_channel=2,
            TU_out_dim=512,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer = TokenGeneratorUnit_single_woScalar(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                                   TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                                   global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                                   TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)
        self.visual = MaskVisionTransformer(pos_embed="learn", embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.linear_head = nn.Linear(TU_out_dim, 128 * 153, bias=True)
        self.predictor = nn.Conv1d(128, 36, kernel_size=1, bias=True)
        self.parcel_index = Get_parcellation()
        self.parcel_index = torch.tensor(
            np.concatenate([self.parcel_index, self.parcel_index + 40962], axis=0)).reshape([-1]).long()
        self.initiate_parameters()

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
        surf = u
        u = self.tokenizer(u)
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
            "u": surf,
            "img_embed_u": x,
            "cls": out
        }


class finetuning_parcellation_SurfClip_single_channel(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            TU_out_dim=512,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer = TokenGeneratorUnit_single(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                                   TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                                   global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                                   TU_dim_head, TU_mlp_dim, TU_dropout, TU_out_dim)
        self.visual = MaskVisionTransformer(pos_embed="learn", embed_dim=TU_out_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.linear_head = nn.Linear(TU_out_dim, 256 * 153, bias=True)
        self.predictor = nn.Conv1d(256, 36, kernel_size=1,bias=True)
        self.parcel_index = Get_parcellation()
        self.parcel_index = torch.tensor(np.concatenate([self.parcel_index, self.parcel_index + 40962], axis=0)).reshape([-1]).long()
        # self.linear_head = nn.Sequential(
        #     nn.Linear(TU_out_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(128, output_channel),
        # )
        self.initiate_parameters()

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
        surf = u
        u = self.tokenizer(u)
        x, attn, ids_restore, tokens, mask = self.encode_image(u,ret=True)
        x_recons = self.linear_head(tokens[:,1:,:])
        out_sum = torch.zeros(x_recons.shape[0], 256, 81924, dtype=x_recons.dtype, device=x_recons.device)
        x_recons = x_recons.reshape(x_recons.shape[0], 640,256,153).permute(0,2,1,3).reshape([x_recons.shape[0],256,-1])
        self.parcel_index = self.parcel_index.to(x_recons.device)
        out_sum.scatter_add_(dim=2, index=self.parcel_index.unsqueeze(0).unsqueeze(0).expand(x_recons.shape[0], 256, -1), src=x_recons)
        counts = torch.bincount(self.parcel_index, minlength=81924)
        out = out_sum/counts
        out = self.predictor(out)

        return {
            "u": surf,
            "img_embed_u": x,
            "cls": out
        }

class finetune_SurfClip(nn.Module):
    def __init__(
            self,
            input_channel, hidden_dim, num_patches, TU_scalar_scales, TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
            global_mean, global_std, TU_transformer_depth, TU_transformer_heads, TU_dim_head, TU_mlp_dim, TU_dropout,
            # Encoder params
            Enc_depth,
            Enc_num_heads,
            output_channel=2,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer = TokenGeneratorUnit(input_channel, hidden_dim, num_patches, TU_scalar_scales,
                                            TU_hidden_dim_scalar_enc, TU_epsilon_scalar_enc,
                                            global_mean, global_std, TU_transformer_depth, TU_transformer_heads,
                                            TU_dim_head, TU_mlp_dim, TU_dropout)
        self.visual = MaskVisionTransformer(pos_embed="learn", embed_dim=hidden_dim, depth=Enc_depth, num_heads=Enc_num_heads)
        self.linear_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_channel),
        )
        self.initiate_parameters()


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
        surf = u
        u = self.tokenizer(u)
        img_embed_u = self.encode_image(u)
        cls = self.linear_head(img_embed_u)
        return {
            "u": surf,
            "img_embed_u": img_embed_u,
            "cls": cls
        }

def CLIP_512(Pretrain = True):
    model = CLIP(global_mean_path="/data2/member/SurfClip/data_surfclip/No_ADHD_mean.npy", global_std_path="/data2/member/SurfClip/data_surfclip/No_ADHD_std.npy", Pretrain=Pretrain)
    return model

def SurfClip_512(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True):
    mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    kwargs={
        "mask_ratio" : mask_ratio
    }
    model = SurfaceCLIP(
        input_channel=6,
        hidden_dim=512,
        num_patches=320,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=mean,
        global_std=std,
        TU_transformer_depth=2,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=4,
        Enc_num_heads=8,
        decoder_depth=1,
        decoder_num_heads=4,
        out_dim=512,
        norm_in_head=None,
        act_in_head="gelu",
        shared_head_teacher= True,
        nlayers=2,
        proj_hidden_dim=256,
        **kwargs
    )
    return model

def SurfClip_512_deeper(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True):
    mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    kwargs={
        "mask_ratio" : mask_ratio
    }
    model = SurfaceCLIP(
        input_channel=6,
        hidden_dim=512,
        num_patches=320,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=mean,
        global_std=std,
        TU_transformer_depth=2,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=6,
        Enc_num_heads=8,
        decoder_depth=1,
        decoder_num_heads=4,
        out_dim=512,
        norm_in_head=None,
        act_in_head="gelu",
        shared_head_teacher= True,
        nlayers=2,
        proj_hidden_dim=256,
        **kwargs
    )
    return model

def SurfClip_512_deeper_10(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True):
    mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    kwargs={
        "mask_ratio" : mask_ratio
    }
    model = SurfaceCLIP(
        input_channel=6,
        hidden_dim=512,
        num_patches=320,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=mean,
        global_std=std,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=10,
        Enc_num_heads=16,
        decoder_depth=1,
        decoder_num_heads=4,
        out_dim=512,
        norm_in_head=None,
        act_in_head="gelu",
        shared_head_teacher= True,
        nlayers=2,
        proj_hidden_dim=256,
        **kwargs
    )
    return model

def SurfClip_512_deeper_10_single_channel(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    kwargs={
        "mask_ratio" : mask_ratio
    }
    model = SurfaceCLIP_single(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        TU_out_dim=512,
        Enc_depth=10,
        Enc_num_heads=16,
        decoder_depth=1,
        decoder_num_heads=4,
        out_dim=512,
        norm_in_head=None,
        act_in_head="gelu",
        shared_head_teacher= True,
        nlayers=2,
        proj_hidden_dim=256,
        **kwargs
    )
    return model

def SurfClip_512_deeper_10_single_channel_woToken(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    kwargs={
        "mask_ratio" : mask_ratio
    }
    model = SurfaceCLIP_single_woTokenizer(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        TU_out_dim=512,
        Enc_depth=10,
        Enc_num_heads=16,
        decoder_depth=1,
        decoder_num_heads=4,
        out_dim=512,
        norm_in_head=None,
        act_in_head="gelu",
        shared_head_teacher= True,
        nlayers=2,
        proj_hidden_dim=256,
        **kwargs
    )
    return model

def SurfClip_512_deeper_10_single_channel_woSpherical(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    kwargs={
        "mask_ratio" : mask_ratio
    }
    model = SurfaceCLIP_single_woSpherical(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        TU_out_dim=512,
        Enc_depth=10,
        Enc_num_heads=16,
        decoder_depth=1,
        decoder_num_heads=4,
        out_dim=512,
        norm_in_head=None,
        act_in_head="gelu",
        shared_head_teacher= True,
        nlayers=2,
        proj_hidden_dim=256,
        **kwargs
    )
    return model


def SurfClip_512_deeper_10_single_channel_woTransformer(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    kwargs={
        "mask_ratio" : mask_ratio
    }
    model = SurfaceCLIP_single_woTransformer(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        TU_out_dim=512,
        Enc_depth=10,
        Enc_num_heads=16,
        decoder_depth=1,
        decoder_num_heads=4,
        out_dim=512,
        norm_in_head=None,
        act_in_head="gelu",
        shared_head_teacher= True,
        nlayers=2,
        proj_hidden_dim=256,
        **kwargs
    )
    return model

def SurfClip_512_deeper_10_single_channel_woScalar(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    kwargs={
        "mask_ratio" : mask_ratio
    }
    model = SurfaceCLIP_single_woScalar(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        TU_out_dim=512,
        Enc_depth=10,
        Enc_num_heads=16,
        decoder_depth=1,
        decoder_num_heads=4,
        out_dim=512,
        norm_in_head=None,
        act_in_head="gelu",
        shared_head_teacher= True,
        nlayers=2,
        proj_hidden_dim=256,
        **kwargs
    )
    return model

def finetuning_SurfClip_512_deeper_10_single_channel(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True,device=None):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    model = finetuning_SurfClip_single_channel(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=10,
        Enc_num_heads=16,
    )
    return model

def finetuning_SurfClip_512_deeper_10_single_channel_woTokenizer(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True,device=None):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    model = finetuning_SurfClip_single_channel_woTokenizer(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=10,
        Enc_num_heads=16,
    )
    return model

def finetuning_SurfClip_512_deeper_10_single_channel_woTransformer(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True,device=None):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    model = finetuning_SurfClip_single_channel_woTransformer(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=10,
        Enc_num_heads=16,
    )
    return model

def finetuning_SurfClip_512_deeper_10_single_channel_woScalar(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True,device=None):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    model = finetuning_SurfClip_single_channel_woScalar(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=10,
        Enc_num_heads=16,
    )
    return model

def finetuning_SurfClip_512_deeper_10_single_channel_woSpherical(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True,device=None):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    model = finetuning_SurfClip_single_channel_woSpherical(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=10,
        Enc_num_heads=16,
    )
    return model

def regression_finetuning_SurfClip_512_deeper_10_single_channel(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True,device=None):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    model = finetuning_SurfClip_single_channel(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=10,
        Enc_num_heads=16,
        output_channel=1
    )
    return model

def Parcellation_finetuning_SurfClip_512_deeper_10_single_channel_woSpherical(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True,device=None):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    model = finetuning_parcellation_SurfClip_single_channel_woSpherical(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=10,
        Enc_num_heads=16
    )
    return model

def Parcellation_finetuning_SurfClip_512_deeper_10_single_channel_woTransformer(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True,device=None):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    model = finetuning_parcellation_SurfClip_single_channel_woTransformer(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=10,
        Enc_num_heads=16
    )
    return model

def Parcellation_finetuning_SurfClip_512_deeper_10_single_channel_woScalar(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True,device=None):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    model = finetuning_parcellation_SurfClip_single_channel_woScalar(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=10,
        Enc_num_heads=16
    )
    return model

def Parcellation_finetuning_SurfClip_512_deeper_10_single_channel(global_mean_path, global_std_path,mask_ratio = 0, Pretrain = True,device=None):
    # mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    # std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    model = finetuning_parcellation_SurfClip_single_channel(
        input_channel=1,
        hidden_dim=256,
        num_patches=640,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=None,
        global_std=None,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=10,
        Enc_num_heads=16
    )
    return model

def finetuneSurfClip_512(global_mean_path, global_std_path, mask_ratio = 0, Pretrain = True):
    mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    model = finetune_SurfClip(
        input_channel=6,
        hidden_dim=512,
        num_patches=320,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=mean,
        global_std=std,
        TU_transformer_depth=2,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=4,
        Enc_num_heads=8,
    )
    return model

def finetuneSurfClip_512_deeper_6(global_mean_path, global_std_path, mask_ratio = 0, Pretrain = True):
    mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    model = finetune_SurfClip(
        input_channel=6,
        hidden_dim=512,
        num_patches=320,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=mean,
        global_std=std,
        TU_transformer_depth=2,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=6,
        Enc_num_heads=8,
    )
    return model

def finetuneSurfClip_512_deeper_10(global_mean_path, global_std_path, mask_ratio = 0, Pretrain = True,device="cuda"):
    mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32)).cuda(device)
    std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32)).cuda(device)
    model = finetune_SurfClip(
        input_channel=6,
        hidden_dim=512,
        num_patches=320,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=mean,
        global_std=std,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=10,
        Enc_num_heads=16,
    )
    return model


def finetuneRegressionSurfClip_512(global_mean_path, global_std_path, mask_ratio = 0, Pretrain = True):
    mean = torch.tensor(np.load(global_mean_path).reshape([1, 6, 1]).astype(np.float32))
    std = torch.tensor(np.load(global_std_path).reshape([1, 6, 1]).astype(np.float32))
    model = finetune_SurfClip(
        input_channel=6,
        hidden_dim=512,
        num_patches=320,
        TU_scalar_scales=None,
        TU_hidden_dim_scalar_enc=32,
        TU_epsilon_scalar_enc=1.1,
        global_mean=mean,
        global_std=std,
        TU_transformer_depth=4,
        TU_transformer_heads=4,
        TU_dim_head=128,
        TU_mlp_dim=256,
        TU_dropout=0.1,
        Enc_depth=10,
        Enc_num_heads=16,
        output_channel=1
    )
    return model

if __name__ == '__main__':
    # model = MaskVisionTransformer(pos_embed="learn",embed_dim=512,depth=6,num_heads=4)
    # model(torch.randn(4,320,512),need_attn=True)    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # attn = torch.randn(size=(4, 320))
    # get_att_mask(attn,ratio=0.5)
    from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler, schedule

    x = torch.randn(10,2,40962).to("cuda:3")
    text = torch.randn(10,512).to("cuda:3")
    encoder = finetuning_SurfClip_512_deeper_10_single_channel(None, None, mask_ratio=0.75).to("cuda:3")
    sch = schedule(wait=1, warmup=1, active=3, repeat=1)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=sch,
                 on_trace_ready=tensorboard_trace_handler("./tb_profile"),
                 record_shapes=True,
                 profile_memory=True,
                 with_stack=False,) as prof:
        torch.cuda.synchronize()
        with record_function("model_inference"):
            out = encoder(x)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


