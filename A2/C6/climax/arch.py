# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from .vision_transformer import PatchEmbed, trunc_normal_ , VariableMapping_Attention

from timm.models.vision_transformer import Block
# for now I am disabling this b.c. I saw discrepancies using only simple_ddp
# it could be because of the extra blocks we add for the forward-decoder
#from .vision_transformer import Block # for now I am disabling this b.c. I saw discrepancies using only simple_ddp

from climax.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)
import torch.distributed as dist

from .parallelpatchembed import ParallelVarPatchEmbed
from einops import rearrange

from .dist_functions import F_Identity_B_Broadcast,F_Broadcast_B_Identity, F_Identity_B_AllReduce




class Pred_Rearrange(nn.Module):
    def __init__(self, aggregated_variables, num_patches, embed_dim):
        super(Pred_Rearrange, self).__init__()
        self.aggregated_variables = aggregated_variables
        self.num_patches = num_patches
        self.embed_dim = embed_dim

    def forward(self, x):
        x=rearrange(x,'b (v l) d -> b l (v d) ',v=self.aggregated_variables,l = self.num_patches, d= self.embed_dim)
        return x

class ClimaX(nn.Module):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
        parallel_patch_embed (bool): whether to use parallel patch embedding
        aggregated_variables (int): the number of aggregated variables
        _ddp_params_and_buffers_to_ignore: what parameters or buffers to ignore for pytorch DDP
        tensor_par_size: the size for the GPU tensor model parallel group
        tensor_par_group: the GPU groups for tensor model parallelism
    """

    def __init__(
        self,
        default_vars,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
        aggregated_variables=1,
        seq_par_size = 1,
        data_par_size = 1,
        tensor_par_size = 1,
        tensor_par_group = None,
        _ddp_params_and_buffers_to_ignore = None,
        mask_ratio = 0.75,
    ):
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        self.parallel_patch_embed = parallel_patch_embed
        self.aggregated_variables = aggregated_variables
        self.seq_par_size = seq_par_size
        self.tensor_par_size = tensor_par_size
        self.data_par_size = data_par_size
        self.tensor_par_group = tensor_par_group
        self.mask_ratio = mask_ratio

        if _ddp_params_and_buffers_to_ignore is not None:
            self._ddp_params_and_buffers_to_ignore =  _ddp_params_and_buffers_to_ignore


        # variable tokenization: separate embedding layer for each input variable
        if self.parallel_patch_embed:
            self.token_embeds = ParallelVarPatchEmbed(len(default_vars), img_size, patch_size, embed_dim)
            self.num_patches = self.token_embeds.num_patches
        else:
            self.token_embeds = nn.ModuleList(
                [PatchEmbed(img_size, patch_size, 1, embed_dim) for i in range(len(default_vars))]
            )
            self.num_patches = self.token_embeds[0].num_patches


        assert ((self.num_patches * self.aggregated_variables)%self.seq_par_size)==0, "(num_patches * aggregted_variables) % sequence parallel size must be 0"

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, aggregated_variables, embed_dim), requires_grad=True)
        self.var_agg = VariableMapping_Attention(embed_dim, num_heads=num_heads, qkv_bias=False, 
                                                 tensor_par_size = tensor_par_size, tensor_par_group = tensor_par_group)

        # positional embedding and lead time embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.lead_time_embed = nn.Linear(1, embed_dim)

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    #drop=drop_rate,
                    #tensor_par_size = tensor_par_size,
                    #tensor_par_group = tensor_par_group,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        decoder_embed_dim=128
        learn_pos_emb=True
        decoder_num_heads=4
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # TODO: each channel has its own mask token

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=learn_pos_emb
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, len(default_vars) * patch_size**2, bias=True
        )  # decoder to token
        # --------------------------------------------------------------------------        

        # prediction head
        #self.head = nn.ModuleList()
        #for _ in range(decoder_depth):
        #    self.head.append(nn.Linear(embed_dim, embed_dim))
        #    self.head.append(nn.GELU())
        #self.head.append(Pred_Rearrange(aggregated_variables=self.aggregated_variables, num_patches=self.num_patches, embed_dim=embed_dim)) 
        #self.head.append(nn.Linear(embed_dim*aggregated_variables, len(self.default_vars) * patch_size**2))
        #self.head = nn.Sequential(*self.head)

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        # token embedding layer
        if self.parallel_patch_embed:
            for i in range(len(self.token_embeds.proj_weights)):
                w = self.token_embeds.proj_weights[i].data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        else:
            for i in range(len(self.token_embeds)):
                w = self.token_embeds[i].proj.weight.data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # initialize decoder components
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = len(self.default_vars)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def unpatchify_vit_mae(self, x):
        """
        x: (N, L, patch_size**2 *C)
        imgs: (N, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        c = x.shape[2] // p**2

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs


    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)

        src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
        x = self.var_agg(var_query, x)  # BxL, V~ , D, where V~ is the aggregated variables
        #x , _ = self.var_agg(var_query, x, x)  # BxL, V~ , D, where V~ is the aggregated variables
        x = x.squeeze()
        x= F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)
        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, V~, D

        if self.aggregated_variables >1:
            x = rearrange(x,'b l v d -> b v l d')

        return x

    def random_masking(self, x, mask_ratio):
        """Perform per-sample random masking by per-sample shuffling.

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

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables,):
        
        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))
            
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # B, V~ , L, D, where V~ is the aggregated variables

        # add pos embedding
        x = x + self.pos_embed      # B, V~ , L ,D

        if self.aggregated_variables >1 :
            x = x.flatten(1, 2)  # B, V~ * L, D

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)  #B, 1, D

        x = x + lead_time_emb  # B, V~ * L, D

        x = self.pos_drop(x)    #A potential issue here for tensor model parallelism, handle carefully

        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)

        dist.broadcast(x, src_rank , group=self.tensor_par_group)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x= F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)
        return x, mask, ids_restore

    def forward_decoder(self, x, variables, ids_restore):

        # embed tokens
        x = self.decoder_embed(x)  # B, L x mask_ratio, D

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle, B, L, D

        # add pos embedding, pos_emb: 1, L, D
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x


    def forward_loss(self, imgs, pred, variables, metric, mask):
        """
        imgs: [B, C, H, W]
        pred: [B, CxL, p*p]
        mask: [B, CxL], 0 is keep, 1 is remove,
        """

        if isinstance(variables, list):
            variables = tuple(variables)
        
        img_mask = mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])  # [B, L, C*p*p]
        img_mask = self.unpatchify(img_mask)[:, 0]  # [B, H, W]

        img_pred = self.unpatchify(pred)  # [B, C, H, W]
        var_ids = self.get_var_ids(variables, imgs.device)
        img_pred = img_pred[:, var_ids]  # only compute loss over present variables
        

        if metric is None:
            return None, img_pred, img_mask

        loss_dict = [m(img_pred, imgs, variables,) for m in metric]

        return loss_dict, img_pred, img_mask

    #def forward(self, imgs, variables, metric, lat, mask_ratio=0.75):
    #    latent, mask, ids_restore = self.forward_encoder(imgs, variables, mask_ratio)
    #    pred = self.forward_decoder(latent, variables, ids_restore)  # [B, L, p*p]
    #    loss, pred, mask = self.forward_loss(imgs, pred, variables, metric, lat, mask)
    #    return loss, pred, mask

    #def pred(self, imgs, variables, mask_ratio):
    #    _, pred, mask = self.forward(imgs, variables, None, None, mask_ratio)
    #    return pred, mask    

    def forward(self, x, lead_times, variables, metric,):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """

        src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)

        dist.broadcast(x, src_rank , group=self.tensor_par_group)

        latent, mask, ids_restore = self.forward_encoder(x, lead_times, variables,)  # B, V~ * L, D

        pred = self.forward_decoder(latent, variables, ids_restore)
        
        loss, pred, mask = self.forward_loss(x, pred, variables, metric, mask)
        return loss, pred, mask

