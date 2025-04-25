#!/bin/bash

# configs/appl.yaml
# appl

# configs/Sentinel-2.yaml
# sentinel2

MASTER_PORT=23465, CUDA_VISIBLE_DEVICES=1,2, \
		   /ccsopen/home/tsarisa/env/miniconda/envs/cucim2/bin/torchrun \
		   --nproc_per_node 2  \
		   --rdzv-endpoint=holly:23465 \
		   train.py \
		   configs/appl_linear.yaml \
		   --fsdp_size 1 \
		   --simple_ddp_size 1 \
		   --seq_par_size 1 \
		   --tensor_par_size 2 \
		   --dataset appl \
		   --arch orbit_linear
