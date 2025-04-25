#!/bin/bash

# channels will only matter when no-real arg
# imagex and imagey if real needs to match the data
# self_attention is dummy arg yet
# args for var is orbit_hier and orbit

for i in {orbit_hier,}
do
    MASTER_PORT=23462, CUDA_VISIBLE_DEVICES=2,3,4,5, \
		       /ccsopen/home/tsarisa/env/miniconda/envs/cucim2/bin/torchrun \
		       --nproc_per_node 4 \
		       --rdzv-endpoint=holly:23462 \
		       ../train.py \
		       ../configs/era5.yaml \
		       --max_epochs 100 \
		       --fa2 \
		       --fsdp_size 1 \
		       --simple_ddp_size 1 \
		       --seq_par_size 1 \
		       --tensor_par_size 4 \
		       --batch_size 512 \
		       --arch $i \
		       --channels 128 \
		       --real \
		       --imagex 32 \
		       --imagey 64 \
		       --embed_dim 512 \
		       --depth 16 \
		       --num_heads 16
    echo "sleeping..."
    sleep 5
    echo "Done"
done
