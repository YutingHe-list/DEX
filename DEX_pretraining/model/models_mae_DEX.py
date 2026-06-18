from functools import partial
from typing import Type

import torch.nn.functional as F
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from .DEXVisionTransformer import DEXBlock

import numpy as np

def resize_patch_embed(state_dict, target_patch_size=16):
    """
    If the checkpoint's patch_embed has a different patch size than the model's patch_embed, resize it using bicubic interpolation.
    This allows loading MAE checkpoints trained with different patch sizes.
    Args:
        state_dict: the checkpoint's state_dict containing the patch_embed weights
        target_patch_size: the patch size of the current model's patch_embed
    Returns:
        state_dict with resized patch_embed weights if needed
    """
    if "patch_embed.proj.weight" not in state_dict:
        return state_dict  # If the checkpoint doesn't have patch_embed weights, just return it as is

    weight = state_dict["patch_embed.proj.weight"]  # [out_c, in_c, H, W]
    out_c, in_c, H, W = weight.shape

    if (H, W) == (target_patch_size, target_patch_size):
        print(f"✅ patch_embed is {target_patch_size}x{target_patch_size}, no need to resize.")
        return state_dict

    print(f"⚠️ Found patch_embed size {H}x{W}, interpolating to {target_patch_size}x{target_patch_size}")

    # Interpolation
    weight_resized = F.interpolate(
        weight, size=(target_patch_size, target_patch_size), mode="bicubic", align_corners=False
    )

    state_dict["patch_embed.proj.weight"] = weight_resized
    return state_dict

def interpolate_pos_embed(state_dict, model):
    """
    If the checkpoint's pos_embed has a different number of patches than the model's pos_embed, resize it using bicubic interpolation.
    This allows loading MAE checkpoints trained with different image sizes or patch sizes.
    Args:
        state_dict: the checkpoint's state_dict containing the pos_embed weights
        model: the current model, used to get the target pos_embed shape
    Returns:
        state_dict with resized pos_embed weights if needed
    """
    # pos_embed shape in checkpoint and model
    if "pos_embed" not in state_dict:
        return state_dict  # If the checkpoint doesn't have pos_embed, just return it as is
    
    pos_embed_ckpt = state_dict["pos_embed"]        # [1, N_ckpt, C]
    pos_embed_model = model              # [1, N_model, C]
    
    # split cls pos and patch pos
    cls_pos_embed = pos_embed_ckpt[:, :1, :]        # [1, 1, C]
    patch_pos_embed = pos_embed_ckpt[:, 1:, :]      # [1, N_patch_ckpt, C]

    # calculate the number of patches in checkpoint and model
    N_patch_ckpt = patch_pos_embed.shape[1]
    H_ckpt = W_ckpt = int(N_patch_ckpt ** 0.5)      # such as 1369 → 37x37
    assert H_ckpt * W_ckpt == N_patch_ckpt, "patch cannot reshape into a square!"

    # reshape into 2D grid
    patch_pos_embed = patch_pos_embed.reshape(1, H_ckpt, W_ckpt, -1).permute(0, 3, 1, 2)  # [1, C, H, W]

    # target size
    N_patch_model = pos_embed_model.shape[1] - 1    # the number of patches in the model
    H_model = W_model = int(N_patch_model ** 0.5)
    assert H_model * W_model == N_patch_model, "the number of patches in the model cannot be reshaped into a square!"

    # interpolate to the target size
    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(H_model, W_model),
        mode="bicubic",
        align_corners=False
    )

    # reshape back to [1, N_patch, C]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, N_patch_model, -1)

    # concatenate class and patch positions
    new_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
    state_dict["pos_embed"] = new_pos_embed

    return state_dict

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



class MaskedAutoencoderViT(nn.Module):
    """ 
    Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self,
                 img_size: int = 224, 
                 patch_size: int = 16, 
                 in_chans: int = 3,
                 min_experts: int = 8,
                 max_experts: int = 8,
                 min_topk: int = 2,
                 max_topk: int = 2,
                 embed_dim: int = 1024, 
                 depth: int = 24, 
                 num_heads: int = 16,
                 decoder_embed_dim: int = 512, 
                 decoder_depth: int = 8, 
                 decoder_num_heads: int = 16,
                 mlp_ratio: float = 4., 
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 norm_pix_loss: bool = False,
                 pretrained=None
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        if min_experts == max_experts:
            self.num_experts_list = [min_experts] * depth
        else:
            self.num_experts_list = np.repeat(
                np.arange(max_experts, min_experts - 1, -1), 
                int(np.ceil(depth / (max_experts - min_experts + 1)))
            )[:depth].tolist()

        if min_topk == max_topk:
            self.top_k_list = [min_topk] * depth
        else:
            self.top_k_list = np.repeat(
                np.arange(min_topk, max_topk + 1, 1), 
                int(np.ceil(depth / (max_topk - min_topk + 1)))
            )[:depth].tolist()
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            DEXBlock(dim=embed_dim, 
                      num_heads=num_heads,
                      num_experts=self.num_experts_list[i],
                      top_k=self.top_k_list[i],
                      mlp_ratio=mlp_ratio, 
                      qkv_bias=True, 
                      init_values= None,
                      norm_layer=norm_layer)
                      for i in range(depth)
                      ])
        
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, 
                                decoder_num_heads, 
                                mlp_ratio, qkv_bias=True, 
                                norm_layer=norm_layer)
                                for i in range(decoder_depth)
                                ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights(pretrained)

    def initialize_weights(self, pretrained):
        if pretrained:
            state_dict = torch.load(pretrained, map_location="cpu")['model']
            for k in list(state_dict.keys()):
                if 'mlp' in k:
                    new_key = k.replace('mlp', 'DEX.director')
                    state_dict[new_key] = state_dict[k]
                    for i in range(self.num_experts_list[int(k.split('.')[1])]):
                        new_key_i = k.replace('mlp', 'DEX.experts.'+str(i))
                        state_dict[new_key_i] = state_dict[k]

                    del state_dict[k]

            state_dict = interpolate_pos_embed(state_dict, self.pos_embed)
            state_dict = resize_patch_embed(state_dict, self.patch_embed.patch_size[0])
            self.load_state_dict(state_dict, strict=False)
            
            # initialization
            # initialize (and freeze) pos_embed by sin-cos embedding
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        else:
            # initialization
            # initialize (and freeze) pos_embed by sin-cos embedding
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

            # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            torch.nn.init.normal_(self.cls_token, std=.02)
            torch.nn.init.normal_(self.mask_token, std=.02)

            # initialize nn.Linear and nn.LayerNorm
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

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
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
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

    def forward_encoder(self, x, mask_ratio, m, noise_std):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        co_losses = []
        bal_losses = []
        for idx, blk in enumerate(self.blocks):
            x, co_loss, bal_loss = blk(x, m=m, noise_std=noise_std, alpha=float(len(self.blocks)-idx))
            co_losses.append(co_loss)
            bal_losses.append(bal_loss)
        x = self.norm(x)

        bal_loss = torch.stack(bal_losses).mean()
        co_loss = torch.stack(co_losses).mean()
        return x, co_loss, bal_loss, mask, ids_restore

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

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, m=0.999, noise_std=0.):
        latent, co_loss, bal_loss, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, m, noise_std)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        MAE_loss = self.forward_loss(imgs, pred, mask)
        return MAE_loss, co_loss, bal_loss, self.unpatchify(pred), mask


def mae_DEX_vit_base(**kwargs):
    model = MaskedAutoencoderViT(
        pretrained='/home/exouser/MINO/model/mae_pretrain_vit_base.pth',
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_DEX_vit_large(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_DEX_vit_huge(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
