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
from timm.models.vision_transformer import Block

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
        decoder_depth=8, # TODO Made 8 from 4 because of Amir's suggestion
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

        self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames # 1

        assert self.t_pred_patch_size > 0, "pred_t_dim must be a multiple of num_frames" + f"({t_patch_size}, {pred_t_dim}, {num_frames})"

        self.patch_embed = patch_embed(
            img_size,  # 224
            patch_size, # 16
            in_chans, # 3
            embed_dim, # 1024
            num_frames, # 16
            t_patch_size, # 1
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
                _num_patches = 1 + num_patches
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

    def random_masking(self, x, mask_ratio_image=0.75, mask_ratio_video=0.9):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim

        if L == 14 ** 2 * 16:
            # video
            mask_ratio = mask_ratio_video
            pass 
        elif L == 14 ** 2 * 1:
            mask_ratio = mask_ratio_image
            # image
            pass
        else:
            print("L is not 3136 (video) or 196 (image)")
            raise NotImplementedError

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
        ids_keep = torch.arange(len_keep).unsqueeze(0).repeat(N, 1).to("cuda") # TODO use the logic of image masking here

        '''
        Returns:
        x_masked: The masked input tensor with the shape (N, len_keep, D), containing the kept elements.
        mask: The binary mask tensor with the shape (N, L), where 0s represent the kept elements and 1s represent the removed elements.
        ids_restore: The indices that would restore the sorted noise tensor to its original order.
        ids_keep: The indices of the first len_keep elements in the sorted noise tensor.
        '''

        return x_masked, mask, ids_restore, ids_keep

    def mask_test_image(self, x):
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

        # Apply the mask to the input tensor
        x_masked = x[mask == 0].view(N, -1, D)
        
        ids_keep = torch.nonzero(mask == 0)[:, 1] # Get only the indices of the kept elements
        ids_remove = torch.nonzero(mask == 1)[:, 1] # Get only the indices of the k elements

        # Create ids_restore by concatenating ids_keep and ids_remove
        ids_restore = torch.cat((ids_keep, ids_remove), dim=0)  
        
        ids_restore = torch.stack([ids_restore] * N)
        ids_keep = torch.stack([ids_keep] * N) # Copy along the batch dimension
        ids_remove = torch.stack([ids_remove] * N) # Copy along the batch dimension
        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio_image, mask_ratio_video, test_spatiotemporal=False, test_temporal=False, test_image=False):
        test_modes = [int(mode) for mode in [test_spatiotemporal, test_temporal, test_image]]
        assert sum(test_modes) <= 1, "Only one or zero test modes can be active at a time"
        
        mask_ratio_image = int(mask_ratio_image * 14 ** 2) / (14 ** 2 * 1) # quantizes it 
        mask_ratio_video = int(mask_ratio_video * 14 ** 2 * 16) / (14 ** 2 * 16) # quantizes it 

        pretraining_mode = True
        if test_spatiotemporal or test_temporal or test_image:
            pretraining_mode = False

        # image x has dimensions torch.Size([1, 3, 1, 224, 224]) 
        # video x has dimensions torch.Size([1, 3, 16, 224, 224]) 

        # embed patches
        # applies a 3D conv that preserves dimensionality but represents information better
        x = self.patch_embed(x)
        N, T, L, C = x.shape # [2, 16, 196, 1024] or [2, 1, 196, 1024]

        x = x.reshape(N, T * L, C) # [2, 3136, 1024] or [2, 196, 1024]

        # masking: length -> length * mask_ratio
        if pretraining_mode:
            # Do random masking
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio_image, mask_ratio_video)
        
        elif test_image:
            x, mask, ids_restore, ids_keep = self.mask_test_image(x)
            
        elif test_spatiotemporal:
            # For videos, mask bottom half of frames 2-16 for spatiotemporal

            # figuring out the index of the masking for this will be a pain
            raise NotImplementedError

        elif test_temporal:
            # For videos, mask frames 9-16 for temporal 
            x, mask, ids_restore, ids_keep = self.mask_temporal(x)
            pass 
        else:
            raise NotImplementedError("Invalid mode. Either have pretraining, test temporal, or test spatiotemporal")
            
        # Check if output is for a video tensor (torch.Size([4, 3136, 1024]))
        # 3136 = 14 ** 2 * 16
        # 196 = 14 ** 2 * 1

        if x.shape[1:] == torch.Size([int(3136*(1-mask_ratio_video)), 1024]) and mask.shape[1:] == torch.Size([3136]):
            # Valid video tensor
            pass
        elif x.shape[1:] == torch.Size([int(196*(1-mask_ratio_image)), 1024]) and mask.shape[1:] == torch.Size([196]):
            # Valid image tensor
            pass
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
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) 
            
            pos_embed += torch.repeat_interleave(
                self.pos_embed_temporal, # (1, 16, 1024)
                self.input_size[1] * self.input_size[2],
                dim=1,
            ) # results in size [1, 14^2 * 16, 1024] regardless of image or video 

            pos_embed = pos_embed.expand(x.shape[0], -1, -1) # copies along batch dimension to match x
            
            # TODO use the same normalization for both videos and images
            
            # TODO try to have less redundant code
            offsets = []
            if ids_keep.shape[1] == (1 - mask_ratio_image) * 196:
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
            elif ids_keep.shape[1] == 196 * 3 / 4:
                # test image
                '''
                Basically, for images, the ids to keep ranges from 0->195, so we need to add 0->15 * 196 to each row of ids_keep
                as if it came from any of the frames (same as above)
                '''
                for batch_index in range(ids_keep.shape[0]):
                    frame_to_simulate = random.randint(0, 15)
                    offset = frame_to_simulate * 196
                    ids_keep[batch_index] = ids_keep[batch_index] + offset 
                    offsets.append(offset)
                
                # TODO use the below once you figure out how to incorporate offsets properly as a tensor into rest of code
                '''
                num_ids_keep = ids_keep.shape[1]
                N = ids_keep.shape[0]

                # Generate random integers for each row (along the first dimension)
                frame_to_simulate = torch.randint(0, 16, size=(N, 1))

                # Calculate the offsets
                offsets = frame_to_simulate * 196

                # Expand the offsets tensor to match ids_keep shape
                offsets_expanded = offsets.expand(N, num_ids_keep).cuda()
                
                # Add offsets to ids_keep
                ids_keep = ids_keep + offsets_expanded
                '''       
            elif ids_keep.shape[1] == (1 - mask_ratio_video) * 3136:
                # video
                pass
            elif ids_keep.shape[1] == 3136 * 9 / 16:
                # video temporal inference
                pass
            else:
                print("got ids_keep shape not supported, probably an unsupported masking ratio or" 
                      + "tried video spatiotemporal inference which isn't supported. got ids keep shape", ids_keep.shape)
                exit()

            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            ) # pos embed only kept for the patches that are not masked

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

    def forward_decoder(self, x, ids_restore, offsets: list, mask_ratio_image=0.75, mask_ratio_video=0.9):
        # TODO update offsets to work w out list
        # if not offsets:
        #     offsets = [0] * x.shape[0]
        
        # assert len(offsets) == x.shape[0], "each offset corresponds to a single batch"
        
        mask_ratio_image = int(mask_ratio_image * 14 ** 2) / (14 ** 2) # quantizes it 
        mask_ratio_video = int(mask_ratio_video * 14 ** 2 * 16) / (14 ** 2 * 16) # quantizes it 

        if x.shape[1] == 14 ** 2 * (1 - mask_ratio_image) * 1 or x.shape[1] == 14 ** 2 * (0.75) * 1: # image and image test. functionally the same
            T = 1 
        elif x.shape[1] == 14 ** 2 * (1 - mask_ratio_video) * 16: # video
            T = 16
        elif x.shape[1] == 3136 * 9 / 16: # video temporal inference
            T = 16
        else:
            raise NotImplementedError("got unsupported x, x shape was " + str(x.shape))
        
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

        # TODO comment from Amir
        # the following code L427-L449 uses loops, rewrite without using loops to make it more GPU efficient
        # The idea is that you almost never want to use for loops for computation because it is inefficient, unless you have too. E.g, SGD.
        # DEBUG check decoder_pos_emb shape
        if x.shape[1] == 1 + 14 ** 2: # image
            # Offset code (random temporal embedding) WARNING do not keep with below
            # for batch_index in range(x.shape[0]):
            #     cls = x[batch_index, :1, :] # store cls on the side because unaffected by frame offsets
            #     data_x = x[batch_index, 1:, :]

            #     cls_decoder_pos_embed = decoder_pos_embed[0, :1, :]
            #     data_decoder_pos_embed = decoder_pos_embed[0, 1:, :]

            #     first_frame = offsets[batch_index]
            #     last_frame = offsets[batch_index] + 196
            #     data_decoder_pos_embed = data_decoder_pos_embed[first_frame:last_frame, :]
                
            #     x[batch_index, 1:, :] = data_x # + data_decoder_pos_embed # WARNING removed pos embedding as per amirs suggestion
            #     x[batch_index, :1, :] = cls # + cls_decoder_pos_embed # WARNING removed pos embedding as per amirs suggestion

            # WARNING do not keep with above
            # No offsets
            x = x + decoder_pos_embed[:, :197, :] # NOTE treats as if came from first frame only
            pass
        elif x.shape[1] == 1 + (14 ** 2) * 16: # video
            x = x + decoder_pos_embed
        else:
            print("got bad x shape when adding decoder pos emb", x.shape)
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
        x = self.decoder_pred(x) # Linear into correct patchified dimensions
        
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
        
        assert torch.allclose(_imgs.float(), self.unpatchify(target).float(), atol=0.01, rtol=0.01), "unpatchify(target) should yield _imgs"
        assert not self.norm_pix_loss
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape) 
        
        assert mask.sum() > 0, "mask should have at least one 1"

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio_image=0.75, mask_ratio_video=0.9, test_spatiotemporal=False, test_temporal=False, test_image=False):
        latent, mask, ids_restore, offsets = self.forward_encoder(imgs, mask_ratio_image, mask_ratio_video, test_spatiotemporal, test_temporal, test_image)
        pred = self.forward_decoder(latent, ids_restore, offsets, mask_ratio_image, mask_ratio_video)  # [N, L, p*p*3]
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