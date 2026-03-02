#!/bin/bash
#SBATCH --partition=alus_a100
#SBATCH --error=./picking_err_%j.txt
#SBATCH --output=./picking_output_%j.txt
#SBATCH --nodelist=node[205]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=achin@rockefeller.edu
#SBATCH --exclusive

echo Starting at `date`
echo This is job $SLURM_JOB_ID
echo Running on `hostname`

for i in {0..1}; do 
    srun --nodes=1 -w node205 /rugpfs/fs0/cem/store/mreynolds/scripts/in_development/particle_picker/particle_picker_k255e.py \
        --semseg_net_dir /rugpfs/fs0/cem/store/achin/fsgs/training_network/do_training/result_trained_networks_FCN/k255e_semSeg_binaryCrossEnt.h5 \
        --binned_micro_dir /rugpfs/fs0/cem/scratch/achin/wtactn4force_final/micrographs_bin4/\*.mrc \
        --hist_match_dir /rugpfs/fs0/cem/store/achin/fsgs/training_network/generate_synth_data/temp_fixedSemSeg/noise_dir/actin_rotated00000.mrc \
        --increment 32 \
        --box_size 128 \
        --fudge_fac 1.2 \
        --threshold 0.8 \
        --gpu_idx $i \
        --tot_gpus 4 \
        --output_dir wtactn4apyrase_picks/ &
done


wait
