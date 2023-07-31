# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import os.path
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import PatchEmbed, Block
from mae_image.pos_embed_image import get_2d_sincos_pos_embed

import sys
sys.path.append("../")
from vqgan import get_vq_model
from X_CLIP.cct import CrossFramelAttentionBlock
from AIM.vit_clip import ResidualAttentionBlock
from util.video_vit import CheckpointVideoBlock, VideoBlock

class CheckpointImageBlock(Block):
    def forward(self, x, T=16):
        norm1_x = self.norm1(x)
        x = x + self.drop_path(checkpoint(self.attn, norm1_x))
        norm2_x = self.norm2(x)
        x = x + self.drop_path(checkpoint(self.mlp, norm2_x))
        return x

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, transfer_encoder_depth=0,
                 transfer_decoder_depth=0, video_encoder_depth=0, 
                 video_decoder_depth=0, mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 X_CLIP=False, AIM=False, **kwargs):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE transfer
        self.X_CLIP = X_CLIP
        self.AIM = AIM
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.embed_dim = embed_dim
        
        self.video_encoder_depth = video_encoder_depth
        if video_encoder_depth > 0:
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, 16, embed_dim)
            )

        if X_CLIP:
            self.blocks = nn.ModuleList([
                # CheckpointImageBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) if i < (depth - transfer_encoder_depth) else CrossFramelAttentionBlock(d_model=embed_dim, n_head=num_heads, mlp_ratio=mlp_ratio)
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) if i < (depth - transfer_encoder_depth) else CrossFramelAttentionBlock(d_model=embed_dim, n_head=num_heads, mlp_ratio=mlp_ratio)
                for i in range(depth)])
            self.video_blocks = nn.ModuleList([
                # CheckpointVideoBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                VideoBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(video_encoder_depth)])
        elif AIM:
            self.blocks = nn.ModuleList([
                CheckpointImageBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) if i < (depth - transfer_encoder_depth) else ResidualAttentionBlock(d_model=embed_dim, n_head=num_heads, mlp_ratio=mlp_ratio, num_frames=16, scale=0.5)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                CheckpointImageBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.vae = get_vq_model().eval()
        vocab_size = 1024

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_embed_dim = decoder_embed_dim
        self.video_decoder_depth = video_decoder_depth
        if video_decoder_depth > 0:
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, 16, decoder_embed_dim)
            )

        if X_CLIP:
            self.decoder_blocks = nn.ModuleList([
                # CheckpointImageBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) if i < (decoder_depth - transfer_decoder_depth) else CrossFramelAttentionBlock(d_model=decoder_embed_dim, n_head=decoder_num_heads, mlp_ratio=mlp_ratio)
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) if i < (decoder_depth - transfer_decoder_depth) else CrossFramelAttentionBlock(d_model=decoder_embed_dim, n_head=decoder_num_heads, mlp_ratio=mlp_ratio)
                for i in range(decoder_depth)])
            self.decoder_video_blocks = nn.ModuleList([
                # CheckpointVideoBlock(decoder_embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                VideoBlock(decoder_embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(video_decoder_depth)])
        elif AIM:
            self.decoder_blocks = nn.ModuleList([
                CheckpointImageBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) if i < (decoder_depth - transfer_decoder_depth) else ResidualAttentionBlock(d_model=embed_dim, n_head=num_heads, mlp_ratio=mlp_ratio, num_frames=16, scale=0.5)
                for i in range(decoder_depth)])
        else:
            self.decoder_blocks = nn.ModuleList([
                CheckpointImageBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, vocab_size, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------
        # Freezing Respect Weights
        if self.AIM:
            for n, p in self.named_parameters():
                if 'Adapter' not in n:
                    p.requires_grad_(False)

        if train_cct_only: #Testing for training only CCT blocks
            grad_blocks = []
            for j in range(transfer_encoder_depth):
                i = (depth - transfer_encoder_depth) + j
                grad_blocks.append('blocks.{i}.'.format(i=i))
                
            for j in range(transfer_decoder_depth):
                i = (depth - transfer_decoder_depth) + j
                grad_blocks.append('decoder_blocks.{i}.'.format(i=i))
                
            for n, p in self.named_parameters():
                requires_grad = False
                for grad_block in grad_blocks:
                    if grad_block in n:
                        requires_grad = True

                p.requires_grad_(requires_grad)
        
        if train_video_only: #Testing for training only video blocks        
            for n, p in self.named_parameters():
                if 'video' not in n and 'temporal' not in n:
                    print('n no grad: ', n)
                    p.requires_grad_(False)
                else: 
                    print("n grad: ", n)
        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        if self.video_encoder_depth > 0:
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        if self.video_decoder_depth > 0:
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
        if self.AIM:   
            ## initialize S_Adapter
            for n, m in self.blocks.named_modules():
                if 'S_Adapter' in n:
                    for n2, m2 in m.named_modules():
                        if 'D_fc2' in n2:
                            if isinstance(m2, nn.Linear):
                                print('n, n2: ', n, n2)
                                nn.init.constant_(m2.weight, 0)
                                nn.init.constant_(m2.bias, 0)
                                
            for n, m in self.decoder_blocks.named_modules():
                if 'S_Adapter' in n:
                    for n2, m2 in m.named_modules():
                        if 'D_fc2' in n2:
                            if isinstance(m2, nn.Linear):
                                print('n, n2: ', n, n2)
                                nn.init.constant_(m2.weight, 0)
                                nn.init.constant_(m2.bias, 0)

            ## initialize T_Adapter
            for n, m in self.blocks.named_modules():
                if 'T_Adapter' in n:
                    for n2, m2 in m.named_modules():
                        if 'D_fc2' in n2:
                            if isinstance(m2, nn.Linear):
                                print('n, n2: ', n, n2)
                                nn.init.constant_(m2.weight, 0)
                                nn.init.constant_(m2.bias, 0)
                                
            for n, m in self.decoder_blocks.named_modules():
                if 'T_Adapter' in n:
                    for n2, m2 in m.named_modules():
                        if 'D_fc2' in n2:
                            if isinstance(m2, nn.Linear):
                                print('n, n2: ', n, n2)
                                nn.init.constant_(m2.weight, 0)
                                nn.init.constant_(m2.bias, 0)

            ## initialize MLP_Adapter
            for n, m in self.blocks.named_modules():
                if 'MLP_Adapter' in n:
                    for n2, m2 in m.named_modules():
                        if 'D_fc2' in n2:
                            if isinstance(m2, nn.Linear):
                                print('n, n2: ', n, n2)
                                nn.init.constant_(m2.weight, 0)
                                nn.init.constant_(m2.bias, 0)
                                
            for n, m in self.decoder_blocks.named_modules():
                if 'MLP_Adapter' in n:
                    for n2, m2 in m.named_modules():
                        if 'D_fc2' in n2:
                            if isinstance(m2, nn.Linear):
                                print('n, n2: ', n, n2)
                                nn.init.constant_(m2.weight, 0)
                                nn.init.constant_(m2.bias, 0)

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
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
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
    
    def video_eval_mask(self, x, type):
        """
        Mask the last 7 frames of the video patches.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch * time (batch=1 for eval), length, dim

        F, H, W = 16, 14, 14  # F is the number of frames
        
        if type == "spatiotemporal":
            condition = lambda row, col, frame: frame >= (F // 2) and row >= (H // 2)
        elif type == "temporal":
            condition = lambda row, col, frame: frame > (F // 2)
        elif type == "frame prediction":
            condition = lambda row, col, frame: frame != 0
        elif type == "frame interpolation":
            condition = lambda row, col, frame: frame != 0 and frame != (F - 1)
        elif type == "central inpainting":
            condition = lambda row, col, frame: (1 < row < 12) and (1 < col < 12)
        elif type == "dynamic inpainting":
            condition = lambda row, col, frame: (1 < row < 12) and (col > frame and col < (frame + 9))
        elif type == "2x2 tube":
            condition = lambda row, col, frame: row >= (W // 2) and col >= (H // 2)
        elif type == "view":
            raise NotImplementedError
        else: 
            raise NotImplementedError

        # Create the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([N, L], device=x.device)
        for frame in range(F):
            for row in range(H):
                for col in range(W):
                    if condition(row, col, frame):
                        mask[frame:, row * W + col] = 1

        x_masked, ids_restore, ids_keep = self.setup_mask(x, mask, N, D)

        return x_masked, mask, ids_restore, ids_keep

    def setup_mask(self, x, mask, N, D):
        #Getting x_masked, ids_restore, and ids_keep
        
        # Apply the mask to the input tensor
        x_masked = x[mask == 0].view(N, -1, D)
        
        ids_keep = torch.nonzero(mask == 0)[:, 1] # Get only the indices of the kept elements
        ids_remove = torch.nonzero(mask == 1)[:, 1] # Get only the indices of the k elements

        # Create ids_restore by concatenating ids_keep and ids_remove
        ids_shuffle = torch.cat((ids_keep, ids_remove), dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)
        
        ids_keep = ids_keep.unsqueeze(0).repeat(N, 1).to(x.device)
        ids_restore = ids_restore.unsqueeze(0).repeat(N, 1).to(x.device)
        
        return x_masked, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio=0.9, test_image=False, video_test_type='', T=16, N=1):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if test_image:
            x, mask, ids_restore, _ = self.mask_test_image_video(x)
        elif video_test_type and video_test_type != '2x2 tube':
            x, mask, ids_restore, _ = self.video_eval_mask(x, video_test_type)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            if self.X_CLIP or self.AIM:
                x = blk(x, T=T)
            else:
                x = blk(x)

        if self.video_encoder_depth > 0:
            patches = x.shape[1]
            x = x.reshape(N, -1, self.embed_dim)
            if T == 16:
                temporal_embed = torch.repeat_interleave(
                    self.pos_embed_temporal,
                    patches,
                    dim=1,
                )
                temporal_embed = temporal_embed.expand(N, -1, -1)
                x += temporal_embed
                
            for blk in self.video_blocks:
                x = blk(x)
            x = x.reshape(N * T, -1, self.embed_dim)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, T=16, N=1):
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
            if self.X_CLIP or self.AIM:
                x = blk(x, T=T)
            else:
                x = blk(x)
            
        if self.video_decoder_depth > 0:
            patches = x.shape[1]
            x = x.reshape(N, -1, self.decoder_embed_dim)
            if T == 16:
                temporal_embed = torch.repeat_interleave(
                    self.decoder_pos_embed_temporal,
                    patches,
                    dim=1,
                )
                temporal_embed = temporal_embed.expand(N, -1, -1)
                x += temporal_embed
            for blk in self.decoder_video_blocks:
                x = blk(x)
            x = x.reshape(N * T, -1, self.decoder_embed_dim)
        
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
        with torch.no_grad():
            target = self.vae.get_codebook_indices(imgs).flatten(1)
        loss = nn.CrossEntropyLoss(reduction='none')(input=pred.permute(0, 2, 1), target=target)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, visual_tokens=None, mask_ratio_image=0.75, mask_ratio_video=0.9, mask_inpt_mask=None, test_image=False, video_test_type=''):
        if self.AIM:
            self.eval()
            
        N, C, T, H, W = imgs.shape
        
        if video_test_type == '2x2 tube':
            test_image = True
        
        imgs = imgs.contiguous().view(N*T, C, H, W)   
        if not video_test_type or video_test_type == '2x2 tube':
            if T == 1:
                mask_ratio = mask_ratio_image
            else:
                mask_ratio = mask_ratio_video
                
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, test_image, T=T, N=N)
        else:
            raise NotImplementedError
        
        pred = self.forward_decoder(latent, ids_restore, T=T, N=N)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        
        pred = pred.contiguous().view(N, -1, 1024) #N*T, -1, 1024
        return loss, pred, mask

    def mask_test_image_video(self, x):
        """
        Mask the bottom right quadrant of the image patches.
        x: [N, L, D], sequence
        keep_ratio: float, percentage of patches to keep (0.0 to 1.0)
        """
        N, L, D = x.shape  # batch, length, dim

        assert L == 196, "This only works for L = 196 (image)"

        H, W = 14, 14

        # Create the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([N, L], device=x.device)
        for row in range(H):
            for col in range(W):
                if row >= (H * 0.5) and col >= (W * 0.5):
                    mask[:, row * W + col] = 1

        x_masked, ids_restore, ids_keep = self.setup_mask(x, mask, N, D)

        return x_masked, mask, ids_restore, ids_keep
    
    def setup_mask(self, x, mask, N, D):
        #Getting x_masked, ids_restore, and ids_keep
        
        # Apply the mask to the input tensor
        x_masked = x[mask == 0].view(N, -1, D)
        
        ids_keep = torch.nonzero(mask[0] == 0)[:, 0] # Get only the indices of the kept elements
        ids_remove = torch.nonzero(mask[0] == 1)[:, 0] # Get only the indices of the k elements

        # Create ids_restore by concatenating ids_keep and ids_remove
        ids_shuffle = torch.cat((ids_keep, ids_remove), dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)
        
        ids_keep = ids_keep.unsqueeze(0).repeat(N, 1).to(x.device)
        ids_restore = ids_restore.unsqueeze(0).repeat(N, 1).to(x.device)
        
        return x_masked, ids_restore, ids_keep


def mae_vit_small_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    # model = MaskedAutoencoderViT(
    #     patch_size=16, embed_dim=1024, depth=24, num_heads=16,
    #     decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    #     mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model = MaskedAutoencoderViT(**kwargs)
    return model


def mae_vit_blank(**kwargs):
    model = MaskedAutoencoderViT(**kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
