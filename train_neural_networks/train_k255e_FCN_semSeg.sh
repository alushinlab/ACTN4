#!/bin/bash
#SBATCH --partition=alus_a100
#SBATCH --error=./FCN_errLog_k255e_%j.txt
#SBATCH --output=./FCN_trainingLog_k255e_%j.txt
#SBATCH --nodelist=node[204]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=achin@rockefeller.edu

echo Starting at `date`
echo This is job $SLURM_JOB_ID
echo Running on `hostname`

for i in {0}; do 
    srun /rugpfs/fs0/cem/store/achin/fsgs/training_network/do_training/train_FCN_for_semseg.py \
	--input_noise_dir /rugpfs/fs0/cem/store/achin/fsgs/training_network/generate_synth_data/temp_fixedSemSeg/noise_dir \
	--input_semseg_dir /rugpfs/fs0/cem/store/achin/fsgs/training_network/generate_synth_data/temp_fixedSemSeg/semMap \
	--trained_DAE_path /rugpfs/fs0/cem/store/achin/fsgs/training_network/do_training/result_trained_networks/k255e_box256_CCC0p9907.h5 \
	--proj_dim 128 \
	--numProjs 10000 \
	--tv_split 90 \
	--lr 0.00001 \
	--patience 3 \
	--epochs 10 \
	--batch_size 16 \
	--gpu_idx 2 \
	--preload_ram False \
	--output_dir ./result_trained_networks_FCN  &
done


wait
