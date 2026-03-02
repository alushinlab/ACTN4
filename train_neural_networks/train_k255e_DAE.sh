#!/bin/bash
#SBATCH --partition=alus_a100
#SBATCH --error=./err_%j.txt
#SBATH --output=./output_%j.txt
#SBATCH --nodelist=node[205]
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=achin@rockefeller.edu

echo Starting at `date`
echo This is job $SLURM_JOB_ID
echo Running on `hostname`

for i in {0}; do 
    srun /rugpfs/fs0/cem/store/achin/fsgs/training_network/do_training/train_DAE_UNet.py \
	--input_noise_dir /rugpfs/fs0/cem/store/achin/fsgs/training_network/generate_synth_data/temp/noise_dir \
	--input_noiseless_dir /rugpfs/fs0/cem/store/achin/fsgs/training_network/generate_synth_data/temp/noiseless_dir \
	--proj_dim 128 \
	--numProjs 200000 \
	--tv_split 90 \
	--lr 0.00005 \
	--patience 3 \
	--epochs 30 \
	--batch_size 16 \
	--gpu_idx 2 \
	--preload_ram False \
	--output_dir ./result_trained_networks  &
done


wait
