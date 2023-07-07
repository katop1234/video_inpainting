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
from vqgan import get_vq_model

from X_CLIP.cct import CrossFrameCommunicationTransformer

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
        t_patch_size=2,
        patch_embed=video_vit.PatchEmbed,
        no_qkv_bias=False,
        sep_pos_embed=True,
        trunc_init=False,
        cls_embed=True,
        pred_t_dim=16,
        X_CLIP=False, #False for regular training, True for training only X-CLIP
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

        self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames # 2 

        assert self.t_pred_patch_size > 0, "pred_t_dim must be a multiple of num_frames" + f"({t_patch_size}, {pred_t_dim}, {num_frames})"

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = patch_embed(
            img_size,  # 224
            patch_size, # 16
            in_chans, # 3
            embed_dim, # 1024
            num_frames, # 16
            t_patch_size, # 2
        )

        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
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
        self.vae = get_vq_model().eval() 
        vocab_size = 1024 * self.patch_embed.t_patch_size 
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE cct specifics
        self.cct = CrossFrameCommunicationTransformer(input_resolution=img_size, patch_size=patch_size, width=12, layers=4, heads=4, output_dim=157) #Temporarily hard code
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
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
        self.decoder_pred = nn.Linear(decoder_embed_dim, vocab_size, bias=True)
        # --------------------------------------------------------------------------

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

        if L == 14 ** 2 * self.patch_embed.t_grid_size:
            mask_ratio = mask_ratio_video
            pass 
        elif L == 14 ** 2 * 1:
            mask_ratio = mask_ratio_image
            pass
        else:
            print("Invalid tensor size")
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


    def video_eval_mask(self, x, type):
        """
        Mask the last 7 frames of the video patches.
        x: [N, L, D], sequence
        """
    
        N, L, D = x.shape  # batch, length, dim

        assert L == 196 * self.patch_embed.t_grid_size, "This only works for L = 196 * 16 (video)"

        F, H, W = self.patch_embed.t_grid_size, 14, 14  # F is the number of frames
        
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
                        mask[:, frame * H * W + row * W + col] = 1

        x_masked, ids_restore, ids_keep = self.setup_mask(x, mask, N, D)

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


    def forward_encoder(self, x, mask_ratio_image, mask_ratio_video, test_image=False, video_test_type= ""):
        test_modes = not test_image or not video_test_type
        assert test_modes
        
        mask_ratio_image = int(mask_ratio_image * 14 ** 2) / (14 ** 2 * 1) # quantizes it 
        mask_ratio_video = int(mask_ratio_video * 14 ** 2 * 16) / (14 ** 2 * 16) # quantizes it

        pretraining_mode = True
        if test_image or video_test_type:
            pretraining_mode = False

        # x .shape ==  (B, C, T, H, W). For image T == 1, for video T > 1
        x = self.patch_embed(x)
        N, T, L, C = x.shape
        x = x.view(N, T * L, C)

        if pretraining_mode:
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio_image, mask_ratio_video)
        elif test_image:
            x, mask, ids_restore, ids_keep = self.mask_test_image(x)
        elif video_test_type:
            x, mask, ids_restore, ids_keep = self.video_eval_mask(x, video_test_type)
        else:
            raise NotImplementedError("Invalid mode.")

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
            
            if T == self.patch_embed.t_grid_size or video_test_type: #8
                #Add Temporal Embedding for Videos, Not for Images
                pos_embed += torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.input_size[1] * self.input_size[2],
                    dim=1,
                )

            pos_embed = pos_embed.expand(x.shape[0], -1, -1) # copies along batch dimension to match x
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

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, mask_ratio_image=0.75, mask_ratio_video=0.9):

        mask_ratio_image = int(mask_ratio_image * 14 ** 2) / (14 ** 2) # quantizes it 
        mask_ratio_video = int(mask_ratio_video * 14 ** 2 * 16) / (14 ** 2 * 16) # quantizes it 

        if x.shape[1] == 14 ** 2 * (1 - mask_ratio_image) * 1 or x.shape[1] == 14 ** 2 * (0.75) * 1: # image and image test
            T = 1 
        else: #Video case
            T = self.patch_embed.t_grid_size
        
        N = x.shape[0]
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        num_unmasked_tokens = x.shape[1]
        mask_tokens = self.mask_token.repeat(N, T * H * W - num_unmasked_tokens, 1)

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

            if x.shape[1] == 1 + (14 ** 2) * self.patch_embed.t_grid_size: #8
                # Add Temporal Embedding for Video only
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

        if x.shape[1] == 1 + 14 ** 2 * 1: # image
            # Create a range tensor for indexing
            index_range = torch.arange(0, 197, device=x.device).view(1, -1)
            x[:, :197] = x[:, :197] + decoder_pos_embed[:, index_range]
        elif x.shape[1] == 1 + (14 ** 2) * self.patch_embed.t_grid_size: # video, 8
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
        pred: [N, t*h*w, u*p*p*3] pred: [N, t*h*w, u*1024] t*h*w ==196 for some reason not sure (u = 1)
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
                .to(imgs.device)
            )
        else:
            # images
            _imgs = imgs

        N = _imgs.shape[0]
        T = _imgs.shape[2]

        with torch.no_grad():  
            _imgs = _imgs.permute(0, 2, 1, 3, 4).flatten(0, 1)
            target = self.vae.get_codebook_indices(_imgs).flatten(1)
            target = torch.reshape(target, [N, T * 196])
        
        pred = pred.view(N, -1, 196, 2, 1024)
        pred = pred.permute(0, 1, 3, 2, 4)
        pred = pred.flatten(1, 2)
        pred = pred.flatten(1, 2)
        
        loss = nn.CrossEntropyLoss(reduction='none')(input=pred.permute(0, 2, 1), target=target)
        loss = (loss * mask).sum() / mask.sum() #mean loss on removed patches
        return loss


    def forward(self, imgs, mask_ratio_image=0.75, mask_ratio_video=0.9, test_image=False, video_test_type=""):
        self.vae.eval()
        if imgs.shape[2] == 1: #images
            repeat = self.patch_embed.t_patch_size
            imgs = imgs.repeat(1, 1, repeat, 1, 1)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio_image, mask_ratio_video, test_image, video_test_type)
        print('latent.shape: ', latent.shape)
        # latent = self.cct(latent)
        pred = self.forward_decoder(latent, ids_restore, mask_ratio_image, mask_ratio_video) #[N, L, 1024]
        mask = mask.repeat_interleave(self.patch_embed.t_patch_size, dim=1)
        
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