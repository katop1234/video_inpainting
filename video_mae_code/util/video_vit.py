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
    def __init__(self, dim_in, dim_out, heads, norm=True, dim_context=None, norm_context=False, **attn_kwargs):
        super().__init__()
        self.cross_attn = rin.CrossAttention(dim_in, dim_context=dim_context, heads=heads, norm=norm, norm_context=norm_context, **attn_kwargs)
        self.ff = rin.FeedForward(dim_out)
        
    def forward(self, x, context=None):
        x_prev = x.clone().detach()  # Store the previous x
        x = self.cross_attn(x, context) + x
        x = self.ff(x) + x
        return x, x_prev

class RINBlockVIP(nn.Module):
    def __init__(
        self,
        dim,
        process_depth=4,
        dim_latent=None,
        final_norm=True,
        heads=16,
        read_depth=1,
        write_depth=1,
        **attn_kwargs
    ):
        super().__init__()
        dim_latent = rin.default(dim_latent, dim)

        self.read_blocks = nn.ModuleList([
            TransformerBlock(dim_latent, dim_latent, heads, dim_context=dim, **attn_kwargs)
            for _ in range(read_depth)
        ])

        self.process_blocks = nn.ModuleList([
            TransformerBlock(dim_latent, dim_latent, heads, **attn_kwargs)
            for _ in range(process_depth)
        ])

        self.write_blocks = nn.ModuleList([
            TransformerBlock(dim, dim, heads, dim_context=dim_latent, norm_context=True, **attn_kwargs)
            for _ in range(write_depth)
        ])

        self.latent_final_norm = rin.LayerNorm(dim_latent) if final_norm else nn.Identity()

        self.counter = 0
        self.print_frequency = 100  # Change this to control how often the similarities are printed

    def forward(self, patches, latents, print_similarities=False):
        # Helper function to calculate and print similarity
        def print_similarity(old, new, block_name, depth):
            similarity = torch.sum(new * old) / (torch.norm(new) * torch.norm(old))
            print(f'{block_name} similarity at depth {depth}: {similarity.item()}')
            
        latents_initial = latents.clone().detach()  # Store the initial latents
        patches_initial = patches.clone().detach()  # Store the initial patches
        
        if self.counter % self.print_frequency == 0:
            print("---Start of RIN Block---")

        for i, read_block in enumerate(self.read_blocks):
            latents_prev = latents.clone().detach()  # Store the previous latents
            latents = read_block[0](latents, patches) + latents
            latents = read_block[1](latents) + latents
            if self.counter % self.print_frequency == 0:
                print_similarity(latents_prev, latents, 'Read latents', i+1)
                
        for i, process_block in enumerate(self.process_blocks):
            latents_prev = latents.clone().detach()  # Store the previous latents
            latents = process_block[0](latents) + latents
            latents = process_block[1](latents) + latents
            if self.counter % self.print_frequency == 0:
                print_similarity(latents_prev, latents, 'Process latents', i+1)

        for i, write_block in enumerate(self.write_blocks):
            patches_prev = patches.clone().detach()  # Store the previous patches
            patches = write_block[0](patches, latents) + patches
            patches = write_block[1](patches) + patches
            if self.counter % self.print_frequency == 0:
                print_similarity(patches_prev, patches, 'Write patches', i+1)

        # Print final similarity values
        if self.counter % self.print_frequency == 0:
            print_similarity(latents_initial, latents, 'Final vs Initial Latent', len(self.read_blocks)+len(self.process_blocks)+len(self.write_blocks))
            print_similarity(patches_initial, patches, 'Final vs Initial Patch', len(self.read_blocks)+len(self.process_blocks)+len(self.write_blocks))
        
        self.counter += 1
        
        latents = self.latent_final_norm(latents)
        
        return patches, latents
    
class FITBlockVIP(nn.Module):
    def __init__(self, dim, G, l, read_depth=1, process_depth=1, write_depth=1, **attn_kwargs):
        super().__init__()
        self.G = G
        self.l = l
        self.latents = nn.Parameter(torch.randn(G, l, dim)) * 0.02

        self.group_attn = rin.CrossAttention(dim, **attn_kwargs)
        self.group_ff = rin.FeedForward(dim)

        self.read_blocks = nn.ModuleList([
            nn.Sequential(
                rin.CrossAttention(dim, dim_context=dim, **attn_kwargs),
                rin.FeedForward(dim)
            ) for _ in range(read_depth)
        ])

        self.process_blocks = nn.ModuleList([
            nn.Sequential(
                rin.CrossAttention(dim, **attn_kwargs),
                rin.FeedForward(dim)
            ) for _ in range(process_depth)
        ])

        self.write_blocks = nn.ModuleList([
            nn.Sequential(
                rin.CrossAttention(dim, dim_context=dim, **attn_kwargs),
                rin.FeedForward(dim)
            ) for _ in range(write_depth)
        ])

    def forward(self, x):
        B, N, _ = x.shape
        x = x.view(B, self.G, -1, x.shape[-1])

        # Step 1: Do self attention within each group
        x = self.group_attn(x)
        x = self.group_ff(x)

        # Step 2: (READ) Each group cross attends to its own latent vectors
        latents_per_group = self.latents.unsqueeze(0).expand(B, -1, -1, -1)
        for read_block in self.read_blocks:
            latents_per_group = read_block[0](latents_per_group, x) + latents_per_group
            latents_per_group = read_block[1](latents_per_group) + latents_per_group

        # Step 3: Concat all the latents
        latents_concat = latents_per_group.view(B, self.G*self.l, -1)

        # Step 4: (PROCESS) Concat all the latents and do self attention globally
        for process_block in self.process_blocks:
            latents_concat = process_block[0](latents_concat) + latents_concat
            latents_concat = process_block[1](latents_concat) + latents_concat

        # Step 5: (WRITE) Write back to x in the reverse process as 2
        latents_per_group = latents_concat.view(B, self.G, self.l, -1)
        for write_block in self.write_blocks:
            x = write_block[0](x, latents_per_group) + x
            x = write_block[1](x) + x

        return x.view(B, N, -1)
