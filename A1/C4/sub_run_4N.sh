#!/bin/bash -l
#SBATCH -A 
#SBATCH -J xvit
#SBATCH -N 4
#SBATCH -t 0:15:00
#SBATCH -q debug
#SBATCH --exclusive
#SBATCH -o logs/xvit.o%j

source /lustre/orion/world-shared/stf218/atsaris/DEEPCAM_2022/new_env/source_env.sh

CMD='
for i in {orbit_token_agg,orbit_linear_token_agg,orbit_linear,}; do
    for j in {2}; do
    	for k in {256}; do
    	    python ../train.py \
       	    ../configs/config.yaml \
       	    --max_epochs 1 \
       	    --fa2 \
       	    --fsdp_size 1 \
       	    --simple_ddp_size 1 \
       	    --seq_par_size 1 \
       	    --tensor_par_size 32 \
       	    --batch_size $j \
       	    --arch $i \
       	    --channels $k \
       	    --imagex 128 \
       	    --imagey 256 \
       	    --embed_dim 8192 \
	    --depth 32 \
	    --num_heads 32
	    
	    echo "sleeping..."
	    sleep 5
	    echo "Done"
	done
    done
done
'

echo $CMD

HOME=/tmp time srun --nodes=${SLURM_NNODES} \
          --ntasks=$((SLURM_NNODES*8)) \
          --ntasks-per-node=8 \
          --gpu-bind=closest \
          -c7 \
          bash -c "$CMD"
