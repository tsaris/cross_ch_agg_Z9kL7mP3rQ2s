#!/bin/bash

# configs/appl.yaml
# appl

# configs/Sentinel-2.yaml
# sentinel2

MASTER_PORT=23464, CUDA_VISIBLE_DEVICES=0, \
		   /ccsopen/home/tsarisa/env/miniconda/envs/cucim2/bin/torchrun \
		   --nproc_per_node 1  \
		   --rdzv-endpoint=holly:23464 \
		   train.py \
		   configs/appl.yaml \
		   --fsdp_size 1 \
		   --simple_ddp_size 1 \
		   --seq_par_size 1 \
		   --tensor_par_size 1 \
		   --dataset appl \
		   --arch orbit
