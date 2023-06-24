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

class RINBlockVIP(nn.Module):
    def __init__(
        self,
        dim,
        process_depth=4,
        dim_latent = None,
        final_norm = True,
        heads=16,
        read_depth=1,
        write_depth=1,
        **attn_kwargs
    ):
        super().__init__()
        dim_latent = rin.default(dim_latent, dim)

        self.read_attn = rin.CrossAttention(dim_latent, dim_context = dim, heads = heads, norm = True, **attn_kwargs)
        self.read_ff = rin.FeedForward(dim_latent)

        self.process_attn = rin.CrossAttention(dim_latent, heads = heads, norm = True, **attn_kwargs)
        self.process_ff = rin.FeedForward(dim_latent)

        self.latent_final_norm = rin.LayerNorm(dim_latent) if final_norm else nn.Identity()

        self.write_attn = rin.CrossAttention(dim, dim_context = dim_latent, heads = heads, norm = True, norm_context = True, **attn_kwargs)
        self.write_ff = rin.FeedForward(dim)
        
        # How often to print statistics
        self.counter = 0  # Add this line to initialize your counter
        self.print_frequency = 100 # Change this to control how often the similarities are printed

        self.read_depth = read_depth
        self.process_depth = process_depth
        self.write_depth = write_depth

    def forward(self, patches, latents):
        # Store a copy of the current vectors to do dot product later with
        latents_previous = latents.clone().detach()
        patches_previous = patches.clone().detach()
        
        # latents extract or cluster information from the patches
        for _ in range(self.read_depth):
            latents = self.read_attn(latents, patches) + latents
            latents = self.read_ff(latents) + latents

        # latent self attention
        for _ in range(self.process_depth):
            latents = self.process_attn(latents) + latents
            latents = self.process_ff(latents) + latents

        # additional cross attention layers
        for _ in range(self.write_depth):
            patches = self.write_attn(patches, latents) + patches
            patches = self.write_ff(patches) + patches
        
        # Calculate and print the dot product/similarity between the current and previous patches
        if self.counter % self.print_frequency == 0:
            similarity = torch.sum(latents * latents_previous) / (torch.norm(latents) * torch.norm(latents_previous))
            similarity_patches = torch.sum(patches * patches_previous) / (torch.norm(patches) * torch.norm(patches_previous))
            print('Latent vector similarity: ', similarity.item(), 'Patch vector similarity: ', similarity_patches.item())
        self.counter += 1
        
        latents = self.latent_final_norm(latents)
        return patches, latents
    
class FITBlockVIP(nn.Module):
    def __init__(self, dim, G, l, read_depth, process_depth, write_depth, heads=16, **attn_kwargs):
        super().__init__()

        self.G = G
        self.l = l
        self.read_depth = read_depth
        self.process_depth = process_depth
        self.write_depth = write_depth

        self.read_attn = rin.CrossAttention(dim, dim_context=dim, heads=heads, norm=True, **attn_kwargs)
        self.read_ff = rin.FeedForward(dim)

        self.process_attn = rin.CrossAttention(dim, heads=heads, norm=True, **attn_kwargs)
        self.process_ff = rin.FeedForward(dim)

        self.write_attn = rin.CrossAttention(dim, dim_context=dim, heads=heads, norm=True, norm_context=True, **attn_kwargs)
        self.write_ff = rin.FeedForward(dim)

        self.latents = nn.Parameter(torch.randn(G, l, dim)) * 0.02 

    def group_attention(self, group):
        group = self.process_attn(group) + group
        group = self.process_ff(group) + group
        return group

    def forward(self, x):
        # Step 1: Divide input into G groups
        groups = x.view(self.G, -1, x.shape[-1])

        # Step 2: Self-attention within each group
        groups = torch.stack([self.group_attention(group) for group in groups])

        # Step 3: Each group cross-attends to its corresponding latent vectors
        for _ in range(self.read_depth):
            for i, group in enumerate(groups):
                self.latents[i] = self.read_attn(self.latents[i], group) + self.latents[i]
                self.latents[i] = self.read_ff(self.latents[i]) + self.latents[i]

        # Step 4: Concat all the latents and do self attention globally
        latents_concat = self.latents.view(-1, x.shape[-1])
        for _ in range(self.process_depth):
            latents_concat = self.process_attn(latents_concat) + latents_concat
            latents_concat = self.process_ff(latents_concat) + latents_concat

        # Update the latents with the globally attended latents
        self.latents = latents_concat.view(self.G, self.l, x.shape[-1])

        # Step 5: Latents for each group write back to that group
        for _ in range(self.write_depth):
            groups = torch.stack([self.write_attn(group, self.latents[i]) + group for i, group in enumerate(groups)])
            groups = torch.stack([self.write_ff(group) + group for group in groups])

        return groups.view(x.shape)


