# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from .vision_transformer import Block, PatchEmbed, trunc_normal_ , VariableMapping_Attention

from climax.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)
import torch.distributed as dist

from .parallelpatchembed import ParallelVarPatchEmbed
from einops import rearrange

from .dist_functions import F_Identity_B_Broadcast,F_Broadcast_B_Identity, F_Identity_B_AllReduce, F_Identity_B_AllReduce_VariableMapping
from .dist_functions import all_gather as mod_all_gather


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
        fa2=False,            
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
        self.var_agg = VariableMapping_Attention(embed_dim, num_heads=num_heads, qkv_bias=False, tensor_par_size = tensor_par_size, tensor_par_group = tensor_par_group)

        self.var_query_per_rank = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_linear_per_rank = torch.nn.Linear(len(default_vars)//self.tensor_par_size, 1)

        self.var_linear_per_rank2 = torch.nn.Linear(64, 1)
        self.var_linear_per_rank3 = torch.nn.Linear(4, 1)        

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
                    drop=drop_rate,
                    tensor_par_size = tensor_par_size,
                    tensor_par_group = tensor_par_group,
                    fa2=fa2,                    
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # --------------------------------------------------------------------------

        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(Pred_Rearrange(aggregated_variables=self.aggregated_variables, num_patches=self.num_patches, embed_dim=embed_dim)) 
        self.head.append(nn.Linear(embed_dim*aggregated_variables, len(self.default_vars) * patch_size**2))
        self.head = nn.Sequential(*self.head)

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
        x = x.squeeze()
        x= F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)
        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, V~, D

        if self.aggregated_variables >1:
            x = rearrange(x,'b l v d -> b v l d')

        return x

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):

        ###### CH PARALLEL ########
        tp_ranks = torch.distributed.get_process_group_ranks(self.tensor_par_group)
        assert (len(variables)%len(tp_ranks))==0, "data-channels % TP ranks must be 0"
        chns_per_tp_rank = len(variables) // len(tp_ranks)
        group_rank = torch.distributed.get_group_rank(self.tensor_par_group, dist.get_rank())
        variables = variables[group_rank*chns_per_tp_rank:group_rank*chns_per_tp_rank + chns_per_tp_rank]
        ###########################

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
        
        ###### CH PARALLEL ########
        x1 = x[:, 0:64, :, :]
        x1 = self.var_linear_per_rank2(x1.transpose(1, 3)).transpose(1, 3)

        x2 = x[:, 64:128, :, :]
        x2 = self.var_linear_per_rank2(x2.transpose(1, 3)).transpose(1, 3)

        x3 = x[:, 128:192, :, :]
        x3 = self.var_linear_per_rank2(x3.transpose(1, 3)).transpose(1, 3)

        x4 = x[:, 192:256, :, :]
        x4 = self.var_linear_per_rank2(x4.transpose(1, 3)).transpose(1, 3)

        x = torch.cat((x1,x2,x3,x4), dim=1)
        x = self.var_linear_per_rank3(x.transpose(1, 3)).transpose(1, 3)

        gathered_tensors = mod_all_gather(x, self.tensor_par_group)
        x = torch.cat(gathered_tensors, dim=1)

        x = self.aggregate_variables(x)
        ###########################
        
        # add pos embedding
        x = x + self.pos_embed

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))
        lead_time_emb = lead_time_emb.unsqueeze(1)

        x = x + lead_time_emb
        
        x = self.pos_drop(x)

        src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)

        dist.broadcast(x, src_rank , group=self.tensor_par_group)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x= F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)
        return x

    def forward(self, x, y, lead_times, variables, out_variables, metric, lat):
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

        handle1= dist.broadcast(y, src_rank , group=self.tensor_par_group,async_op = True )

        out_transformers = self.forward_encoder(x, lead_times, variables)  # B, V~ * L, D

        #print("out_transformers.shape ",out_transformers.shape,flush=True)
        preds = self.head(out_transformers)  # B, L, default(V) *p*p

        preds = self.unpatchify(preds)
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        preds = preds[:, out_var_ids]

        handle1.wait()

        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]

        return loss, preds

    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, clim, log_postfix):
        _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat)
        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]
