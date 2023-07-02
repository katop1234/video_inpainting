# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from util import logging
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp

logger = logging.get_logger(__name__)

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        # temporal related:
        frames=16,
        t_patch_size=1,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert frames % t_patch_size == 0
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (frames // t_patch_size)
        )
        self.input_size = (
            frames // t_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        print("in patch embed",
            f"img_size {img_size} patch_size {patch_size} frames {frames} t_patch_size {t_patch_size}"
        )
        self.img_size = img_size
        self.patch_size = patch_size

        self.frames = frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0]
        self.t_grid_size = frames // t_patch_size

        self.embed_dim = embed_dim

        kernel_size = [t_patch_size] + list(patch_size) # 1, 16, 16
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
 
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        assert T == self.frames or T == 1
        x = self.proj(x)
        x = x.flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        return x # Shape [B=2, T=16, num_patches=196, Embed_dim=1024]

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x

class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
### RIN Implementation below ###
import util.rin as rin

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=16, **attn_kwargs):
        super().__init__()

        self.cross_attention = rin.CrossAttention(dim, heads=heads, norm=True, **attn_kwargs)
        self.feed_forward = rin.FeedForward(dim)
        
    def call_cross_attention(self, x, context=None):
        return self.cross_attention(x, context) if x.shape[1] > 1 else x  # Can't do attention with only one patch

    def call_feed_forward(self, x):
        return self.feed_forward(x)

    def forward(self, x, context=None):
        x = self.call_cross_attention(x, context) + x
        x = self.call_feed_forward(x) + x
        return x

class RINBlockVIP(nn.Module):
    def __init__(
        self,
        dim,
        process_depth=1,
        dim_latent = None,
        final_norm = True,
        heads=16,
        read_depth=1,
        write_depth=1,
        **attn_kwargs
    ):
        super().__init__()
        dim_latent = rin.default(dim_latent, dim) # WARNING we use the same dim for everything.

        self.read_blocks = nn.ModuleList([
            TransformerBlock(dim_latent, **attn_kwargs)
            for _ in range(read_depth)
        ])
        
        self.process_blocks = nn.ModuleList([
            TransformerBlock(dim_latent, **attn_kwargs)
            for _ in range(process_depth)
        ])

        self.write_blocks = nn.ModuleList([
            TransformerBlock(dim_latent, **attn_kwargs)
            for _ in range(write_depth)
        ])

        self.latent_final_norm = rin.LayerNorm(dim_latent) if final_norm else nn.Identity()

        self.counter = 0
        self.print_frequency = 100  # Change this to control how often the similarities are printed
    
    # Helper function to calculate and print similarity
    def _print_similarity(self, old, new, block_name, depth):
        similarity = torch.sum(new * old) / (torch.norm(new) * torch.norm(old))
        print(f'{block_name} similarity at depth {depth}: {similarity.item()}')

    def forward(self, patches, latents, print_similarities=False):
        latents_initial = latents.clone().detach() 
        patches_initial = patches.clone().detach()
        
        if self.counter % self.print_frequency == 0:
            print("---Start of RIN Block---")

        for i, read_block in enumerate(self.read_blocks):
            latents_prev = latents.clone().detach()
            latents = read_block(latents, patches)
            if self.counter % self.print_frequency == 0:
                self._print_similarity(latents_prev, latents, 'Read latents', i+1)
                
        for i, process_block in enumerate(self.process_blocks):
            latents_prev = latents.clone().detach()
            latents = process_block(latents)
            if self.counter % self.print_frequency == 0:
                self._print_similarity(latents_prev, latents, 'Process latents', i+1)

        for i, write_block in enumerate(self.write_blocks):
            patches_prev = patches.clone().detach() 
            patches = write_block(patches, latents)
            if self.counter % self.print_frequency == 0:
                self._print_similarity(patches_prev, patches, 'Write patches', i+1)

        # Print final similarity values
        if self.counter % self.print_frequency == 0:
            self._print_similarity(latents_initial, latents, 'Final vs Initial Latent', len(self.read_blocks)+len(self.process_blocks)+len(self.write_blocks))
            self._print_similarity(patches_initial, patches, 'Final vs Initial Patch', len(self.read_blocks)+len(self.process_blocks)+len(self.write_blocks))
        
        self.counter += 1
        
        latents = self.latent_final_norm(latents)
        
        return patches, latents

class FITBlockVIP(nn.Module):
    def __init__(self, dim, read_depth=1, process_depth=1, write_depth=1, **attn_kwargs):
        super().__init__()
        self.group_size = 196 // 4 # Patches per group
        self.l = self.group_size // 4 # Num latents per group

        self.dim_latent = dim

        self.group_block = TransformerBlock(dim, **attn_kwargs)

        self.read_blocks = nn.ModuleList([
            TransformerBlock(dim, **attn_kwargs)
            for _ in range(read_depth)
        ])

        self.process_blocks = nn.ModuleList([
            TransformerBlock(dim, **attn_kwargs)
            for _ in range(process_depth)
        ])

        self.write_blocks = nn.ModuleList([
            TransformerBlock(dim, **attn_kwargs)
            for _ in range(write_depth)
        ])

        self.print_frequency = 100
        self.counter = 0
        
    def _print_similarity(self, old, new, block_name, depth):
        similarity = torch.sum(new * old) / (torch.norm(new) * torch.norm(old))
        print(f'{block_name} similarity at depth {depth}: {similarity.item()}')

    def forward(self, x, latents, print_similarities=False):
        B, N, D = x.shape
        x_initial = x.clone().detach()

        # calculate the number of groups and leftovers
        G = N // self.group_size
        leftover = N % self.group_size
        
        print("leftover is", leftover)
        exit()

        # Latents per group
        L = self.l

        grouped_count = G * self.group_size

        x_grouped = x[:, :grouped_count, :]
        x_leftover = x[:, grouped_count:, :]
        latents = latents[:, :G*L, :] if leftover else latents # only want latents corresponding to groups

        # Step 1: (GROUP) Each group attends to itself
        x_grouped = x_grouped.reshape(B*G, self.group_size, D)
        x_grouped = self.group_block(x_grouped)
        x_leftover = self.group_block(x_leftover) if leftover else x_leftover

        # Step 2: (READ) Each group cross attends to its own latent vectors
        latents = latents.reshape(B*G, L, D)
        for i, read_block in enumerate(self.read_blocks):
            latents = read_block(latents, x_grouped)
            x_leftover = read_block(x_leftover) if leftover else x_leftover

        # Step 3: (PROCESS) Concat all the latents and do self attention globally
        latents = latents.reshape(B, G*L, D)
        x_leftover = x_leftover.reshape(B, leftover, D)
        latents_concat = torch.cat((latents, x_leftover), dim=1)
        for i, process_block in enumerate(self.process_blocks):
            latents_concat = process_block(latents_concat)

        # Step 4: (WRITE) Write back to x in the reverse process as 2
        latents, x_leftover = latents_concat[:, :G*L, :], latents_concat[:, G*L:, :]
        latents = latents.reshape(B*G, L, D)
        for i, write_block in enumerate(self.write_blocks):
            x_grouped = write_block(x_grouped, latents)
            x_leftover = write_block(x_leftover) if leftover else x_leftover
        
        x_grouped = x_grouped.reshape(B, G*self.group_size, D)
        x = torch.cat((x_grouped, x_leftover), dim=1)

        return x, latents
