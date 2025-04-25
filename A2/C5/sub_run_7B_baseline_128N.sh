#!/bin/bash -l
#SBATCH -A 
#SBATCH -J xvit
#SBATCH -N 128
#SBATCH -t 0:30:00
#SBATCH -q debug
#SBATCH --exclusive
#SBATCH -o logs/xvit.o%j


source /lustre/orion/world-shared/stf218/atsaris/DEEPCAM_2022/new_env/source_env.sh

# --real
# HERE "cuda max reserved memory" "max reserved percentage" MUST
# total_params before FSDP tensor

CMD='
for i in {orbit,}; do
    for j in {2,}; do
    	for k in {1,}; do
    	    python ../train.py \
	    ../configs/appl.yaml \
	    --max_epochs 1 \
	    --fsdp_size 2 \
	    --simple_ddp_size 64 \
	    --seq_par_size 1 \
	    --tensor_par_size 8 \
	    --dataset appl \
	    --arch $i \
	    --batch_size $j \
	    --embed_dim 4096 \
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

# nodes=${SLURM_NNODES}
# ntasks=$((SLURM_NNODES*8))
# ntasks-per-node=8

HOME=/tmp time srun --nodes=${SLURM_NNODES} \
          --ntasks=$((SLURM_NNODES*8)) \
          --ntasks-per-node=8 \
          --gpu-bind=closest \
          -c7 \
          bash -c "$CMD"
