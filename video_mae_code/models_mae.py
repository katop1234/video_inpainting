# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from util import video_vit
import random
from util.logging import master_print as print

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        num_frames=16,
        t_patch_size=1,
        patch_embed=video_vit.PatchEmbed,
        no_qkv_bias=False,
        sep_pos_embed=False,
        trunc_init=False,
        cls_embed=False,
        pred_t_dim=16,
        **kwargs,
    ):

        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.pred_t_dim = pred_t_dim

        # t_patch_size is how many consecutive video frames are grouped together to form a single temporal patch
        # pred_t_dim is how many consecutive temporal patches are predicted
        # num_frames is the total number of video frames in input (16)
        # t_pred_patch_size determines the size of the predicted temporal patches in the output video

        self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames

        assert self.t_pred_patch_size > 0, "pred_t_dim must be a multiple of num_frames" + f"({t_patch_size}, {pred_t_dim}, {num_frames})"

        self.patch_embed = patch_embed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )

        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size # 16, 14, 14

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim) # (1, 16, 1024)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.blocks = nn.ModuleList(
            [
                video_vit.Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                video_vit.Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.t_pred_patch_size * patch_size**2 * in_chans,
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print("model initialized new code")

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Converts a video of several frames (imgs) into patches
    def patchify(self, imgs):
        """
        imgs: (N, 3, T, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.t_pred_patch_size

        print("models_mae images shape: ", imgs.shape, H, W, p, T, u)

        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, 3, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 3))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    # reverts the patchify operation
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, 3))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 3, T, H, W))
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

        '''
        Returns:
        x_masked: The masked input tensor with the shape (N, len_keep, D), containing the kept elements.
        mask: The binary mask tensor with the shape (N, L), where 0s represent the kept elements and 1s represent the removed elements.
        ids_restore: The indices that would restore the sorted noise tensor to its original order.
        ids_keep: The indices of the first len_keep elements in the sorted noise tensor.
        '''

        return x_masked, mask, ids_restore, ids_keep
    
    def mask_spatiotemporal(self, x):
        raise NotImplementedError

    def mask_temporal(self, x):
        """
        Perform fixed masking for frames 10-16.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        
        len_keep = L * 9 // 16

        # Keep the frames 1-9
        x_masked = x[:, :len_keep, :]

        # Create the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # The indices that would restore the sorted noise tensor to its original order
        ids_restore = torch.arange(L, device=x.device).repeat(N, 1).to("cuda")

        # The indices of the first len_keep elements in the sorted noise tensor
        ids_keep = torch.arange(len_keep).unsqueeze(0).repeat(N, 1).to("cuda")

        '''
        Returns:
        x_masked: The masked input tensor with the shape (N, len_keep, D), containing the kept elements.
        mask: The binary mask tensor with the shape (N, L), where 0s represent the kept elements and 1s represent the removed elements.
        ids_restore: The indices that would restore the sorted noise tensor to its original order.
        ids_keep: The indices of the first len_keep elements in the sorted noise tensor.
        '''

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio, test_spatiotemporal=False, test_temporal=False):

        assert not (test_spatiotemporal and test_temporal)

        pretraining_mode = True
        if test_spatiotemporal or test_temporal:
            pretraining_mode = False
        

        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        if pretraining_mode:
            # Do random masking
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio) # TODO try diff mask ratio for image and video

        elif test_spatiotemporal:
            # For videos, mask bottom half of frames 2-16 for spatiotemporal

            # figuring out the index of the masking for this will be a pain
            raise NotImplementedError

        elif test_temporal:
            # For videos, mask frames 9-16 for temporal 
            x, mask, ids_restore, ids_keep = self.mask_temporal(x)
            pass 
        else:
            assert 1==0, "Invalid mode."

        print("x.shape: ", x.shape, "mask shape", mask.shape, "mask ratio", mask_ratio)

        # Check if output is for a video tensor (torch.Size([4, 3136, 1024]))
        if x.shape[1:] == torch.Size([int(3136*(1-mask_ratio)), 1024]) and mask.shape[1:] == torch.Size([3136]):
            print("Output is for a video tensor.")
        elif x.shape[1:] == torch.Size([int(196*(1-mask_ratio)), 1024]) and mask.shape[1:] == torch.Size([196]):
            print("Output is for an image tensor.")
        else:
            if pretraining_mode:
                raise ValueError("The output tensor shapes do not match expected shapes for video or image tensors.")

        x = x.view(N, -1, C)

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:

            # self.pos_embed_spatial.shape is torch.Size([1, 14^2 = 196, 1024]). I.e. each patch has a 1024 dimensional embedding
            pos_embed = self.pos_embed_spatial.repeat( # This line repeats the spatial position embeddings along the temporal dimension 
                1, self.input_size[0], 1
            ) # results in size [1, 16 * 14^2, 1024]
            
            pos_embed += torch.repeat_interleave(
                self.pos_embed_temporal, # (1, 16, 1024)
                self.input_size[1] * self.input_size[2],
                dim=1,
            ) # results in size [1, 14^2 * 16, 1024]

            pos_embed = pos_embed.expand(x.shape[0], -1, -1) # expands the position embeddings tensor to match the batch size of the input tensor x

            offsets = []
            if ids_keep.shape[1] == 49:
                # image
                '''
                Basically, for images, the ids to keep ranges from 0->195, so we need to add 0->15 * 196 to each row of ids_keep
                as if it came from any of the frames
                '''
                for batch_index in range(ids_keep.shape[0]):
                    frame_to_simulate = random.randint(0, 15)
                    offset = frame_to_simulate * 196
                    ids_keep[batch_index] = ids_keep[batch_index] + offset 
                    offsets.append(offset)

            elif ids_keep.shape[1] == 784:
                # video
                pass
            elif ids_keep.shape[1] == 1764:
                # video temporal inference
                pass
            else:
                print("kept support for 75% masking only, can change that all later. got", ids_keep.shape)
                raise NotImplementedError

            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            ) # pos embed for the patches that are not masked

            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )

        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore, offsets

    def forward_decoder(self, x, ids_restore, offsets=[0, 0, 0, 0]):

        if x.shape[1] == 14 ** 2 * (0.25) * 1: # image
            T = 1 
        elif x.shape[1] == 14 ** 2 * (0.25) * 16: # video
            T = 16
        elif x.shape[1] == 3136 * 9 // 16: # video temporal inference
            T = 16
        else:
            print("WARNING ONLY 75% MASKING SUPPORTED. got", x.shape)
            raise NotImplementedError
        
        N = x.shape[0]
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        num_unmasked_tokens = x.shape[1]
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - num_unmasked_tokens, 1)

        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])

        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle

        x = x_.view([N, T * H * W, C])
        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            )
            decoder_pos_embed += torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        if x.shape[1] == 1 + 196: # image
            for batch_index in range(x.shape[0]):
                cls = x[batch_index, :1, :] # store cls on the side because unaffected by frame offsets
                data_x = x[batch_index, 1:, :]

                cls_decoder_pos_embed = decoder_pos_embed[0, :1, :]
                data_decoder_pos_embed = decoder_pos_embed[0, 1:, :]

                first_frame = offsets[batch_index]
                last_frame = offsets[batch_index] + 196
                data_decoder_pos_embed = data_decoder_pos_embed[first_frame:last_frame, :]
                
                x[batch_index, 1:, :] = data_x + data_decoder_pos_embed
                x[batch_index, :1, :] = cls + cls_decoder_pos_embed

        elif x.shape[1] == 1 + 196*16: # video
            x = x + decoder_pos_embed
        else:
            print("no support for masking ratio other than 75%. got", x.shape)
            raise NotImplementedError 
    
        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, T, H, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        if imgs.shape[2] == 16:
            # video
            _imgs = torch.index_select(
                imgs,
                2,
                torch.linspace(
                    0,
                    imgs.shape[2] - 1,
                    self.pred_t_dim,
                )
                .long()
                .to(imgs.device),
            )
        else:
            # images
            _imgs = imgs
            
        target = self.patchify(_imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, test_spatiotemporal=False, test_temporal=False):
        latent, mask, ids_restore, offsets = self.forward_encoder(imgs, mask_ratio, test_spatiotemporal, test_temporal)
        pred = self.forward_decoder(latent, ids_restore, offsets)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def mae_vit_huge_patch14(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
