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
    
### Instant-Attention Implementation below ###

### RIN Implementation below ###
import util.rin as rin, math

class InstantAttnBlock(nn.Module):
    # TODO can even project the input down to smaller dim (i.e. 1024/16), and then cross attention to each block that way
    # because attn is only dependent on D, we can make it super small.
    def __init__(self, dim=1024, context_length=1024, seeker_depth=4, object_dimensionality=1):
        
        '''
        object_dimensionality: 1 for 1D text/audio, 2 for 2D images, 3 for 3D video, etc.
        '''
        
        self.dim = dim
        self.context_length = context_length
        self.seeker_depth = seeker_depth
        self.object_dimensionality = object_dimensionality
        
        self.N_max = self.context_length # finest resolution
        self.N_min = 2 # coarsest resolution
        self.hierarchy_height = 16
        assert self.dim % self.hierarchy_height == 0, "dim should be divisible by hierarchy_height so that hierarchy_dimension is an integer"
        self.hierarchy_dimension = self.dim // self.hierarchy_height
        
        self.seeker_heads = 16
        self.num_locations = int(context_length ** 0.5) # TODO try log(context_length) also
        self.heads = 16
        self.num_read_layers = 4
        self.depth = 8
        
        self.hierarchy = self.create_hierarchy()
        self.hierarchy_pos_embds = self.create_hierarchy_pos_embds()
        self.k_seeker, self.q_seeker, self.v_seeker = self.create_seeker(), self.create_seeker(), self.create_seeker()
        
        self.layers = nn.Sequential()
        for _ in self.depth:
            attn = rin.CrossAttention(self.dim, heads = self.heads, norm = True)
            ff = rin.FeedForward(self.dim)
            self.layers.append(attn)
            self.layers.append(ff)
    
    def create_hierarchy(self):
        # TODO make this work for any dimensions (1d text, 2d images, 3d video, etc.)
        module_list = nn.ModuleList()

        # Iterate over each power of 2, up to context_length // 2
        k = 1
        # TODO actually do this instant-ngp style, where you dont have a fixed growth factor, but it's determined
        # by max_resolution N_max, and Max. entries per level
        while 2**k <= self.context_length // 2:
            # Create a nn.ModuleList with 2^k vectors
            # last token is CLS for that layer
            
            vector_list = nn.ModuleList([nn.Parameter(torch.empty(self.dim).normal_(std=0.02)) for _ in range(2**k + 1)])

            # Add the list to the main module_list
            module_list.append(vector_list)

            k += 1
        
        self.num_hierarchy_levels = k - 1

        return module_list
    
    def create_hierarchy(self):
        L = self.hierarchy_height
        b = math.exp((math.log(self.N_max) - math.log(self.N_min)) / (L-1)) # growth factor
        
    
    def create_hierarchy_pos_embds(self):
        # TODO rewrite this function once getting actual hierarchy is fixed
        module_list = nn.ModuleList()

        # Iterate over each power of 2, up to context_length // 2
        k = 1
        while 2**k <= self.context_length // 2:
            # Create a nn.ModuleList with 2^k positional embeddings
            pos_embd_list = nn.ModuleList([nn.Parameter(torch.empty(self.dim).normal_(std=0.02)) for _ in range(2**k + 1)])

            # Add the list to the main module_list
            module_list.append(pos_embd_list)

            k += 1

        return module_list
    
    def get_hierarchy_shape(self):
        return [len(sublist) for sublist in self.hierarchy]
    
    def flatten_hierarchy(self, hierarchy):
        return [param for sublist in hierarchy for param in sublist]

    def deflatten_hierarchy(self, flattened):
        hierarchy_shape = self.get_hierarchy_shape()
        hierarchy = nn.ModuleList()
        start_idx = 0
        for shape in hierarchy_shape:
            end_idx = start_idx + shape
            sublist = nn.ModuleList(flattened[start_idx:end_idx])
            hierarchy.append(sublist)
            start_idx = end_idx
        return hierarchy

    def add_hierarchy_and_pos_embds(self):
        added_hierarchy = nn.ModuleList()

        # Iterate over each level in the hierarchy
        for hierarchy_level, pos_embd_level in zip(self.hierarchy, self.hierarchy_pos_embds):
            # Add the vectors and positional embeddings at this level
            added_level = nn.ModuleList([vector + pos_embd for vector, pos_embd in zip(hierarchy_level, pos_embd_level)])

            # Add the list to the main module_list
            added_hierarchy.append(added_level)

        return added_hierarchy

    def create_seeker(self):
        seeker_layers = []
        for _ in range(self.seeker_depth):
            seeker_layers.append(rin.CrossAttention(self.dim, heads=self.seeker_heads, norm=True))
            seeker_layers.append(rin.FeedForward(self.dim))
        
        # The final layer maps to w points and applies a sigmoid to get values in [0, 1]
        seeker_layers.append(nn.Sequential(
            nn.Linear(self.dim, self.num_locations),  
            nn.Sigmoid()  # Ensures output is in the range [0, 1]
        ))
        return nn.ModuleList(seeker_layers)

    def get_cls_tokens(self, hierarchy):
        # Get the CLS token from each level of the hierarchy
        CLS_tokens = [level[-1] for level in hierarchy]
        
        # Concatenate the CLS tokens from each level
        CLS_tokens = torch.cat(CLS_tokens, dim=0)
        
        # Add a batch dimension
        CLS_tokens = CLS_tokens.unsqueeze(0)
        
        return CLS_tokens
    
    def get_locations_from_seeker(self, CLS_tokens, seeker):
        for layer in seeker[:-1]:
            CLS_tokens = layer(CLS_tokens) + CLS_tokens
        locations = seeker[-1](CLS_tokens)
        return locations
    
    def get_locations_from_seeker(self, CLS_tokens):
        k_locations = self.get_locations_from_seeker(CLS_tokens, self.k_seeker)
        q_locations = self.get_locations_from_seeker(CLS_tokens, self.q_seeker)
        v_locations = self.get_locations_from_seeker(CLS_tokens, self.v_seeker)
        return k_locations, q_locations, v_locations
        
    def forward(self, x):
        
        hierarchy = self.add_hierarchy_and_pos_embds()
        
        latents = self.flatten_hierarchy(hierarchy)
        
        ### READ ###
        chunk_size = max(1, int(math.log2(len(latents))))
        for _ in range(self.num_read_layers):
            new_latents = torch.zeros_like(latents)
            for i in range(0, len(latents), chunk_size):
                # WARNING due to floating point arithmetic, adding these chunked numbers may be
                # minutely different from doing the original matrix multplication at once
                chunk = latents[i:i+chunk_size]
                updated_chunk = self.read_attn(chunk, x) + chunk
                updated_chunk = self.read_ff(updated_chunk) + updated_chunk
                new_latents[i:i+chunk_size] = updated_chunk
            latents = new_latents
        
        hierarchy = self.deflatten_hierarchy(latents)
        
        # TODO when you try to read from the hierarchy's CLS tokens all concatenated together, you can also
        # concat a context token directly from x, which has no position (just represents whole thing). It is the 𝜉 in NGP
        
        ### PROCESS ###
        CLS_tokens = self.get_cls_tokens(hierarchy)
        
        # Get attention locations
        # TODO can you get 4 seperate sets of attention locations and do them in parallel?
        # TODO after doing these 4 separately, you can them cross attend them all onto x over many 
        # layers. Order can be nn1, nn2, nn3, nn4, nn1... total write_layers number of times. See if parallelizable or too much memory.
        k_locations, q_locations, v_locations = self.get_locations_from_seeker(CLS_tokens)
        
        # TODO remember to concat the vectors from each level of the hierarchy to get the final K Q Vs
        
        # TODO get matrices for K Q V thru the locations above using hierarchy
        
        K = ...
        Q = ...
        V = ...
        
        # TODO what is this supposed to output?
        # how do we want to process the keys and values
        for layer in self.layers:
            x = layer(x) + x
        
        ### WRITE ###
        # TODO once you process the latents, you'll probably cross attend back to x if this is generative.
        # however, if you're doing autoregressive, you don't need to write back to length N, and can just output a prediction  
        # So keep the option to change up what to do after the latents are fully processed
        
        return x

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
