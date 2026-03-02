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

for i in {0..1}; do 
    srun /rugpfs/fs0/cem/store/achin/fsgs/training_network/generate_synth_data/projection_generator_guiVersion.py \
        --input_mrc_dir /rugpfs/fs0/cem/store/achin/fsgs/training_network/input_mrcs \
	--output_noise_dir /rugpfs/fs0/cem/store/achin/fsgs/training_network/generate_synth_data/temp_fixedSemSeg/noise_dir \
	--output_noiseless_dir /rugpfs/fs0/cem/store/achin/fsgs/training_network/generate_synth_data/temp_fixedSemSeg/noiseless_dir \
	--output_semMap_dir /rugpfs/fs0/cem/store/achin/fsgs/training_network/generate_synth_data/temp_fixedSemSeg/semMap \
	--numProjs 10032 \
	--nProcs 48 \
	--box_len 128 \
	--alt_stdev 10 \
	--tx_low -50 \
	--tx_high 50 \
	--ty_low -50 \
	--ty_high 50 \
	--tz_low -50 \
	--tz_high 50 \
	--max_fil_num 3 \
	--is_bundle False \
	--kV 300 \
	--ampCont 10.0 \
	--angpix 4.36 \
	--bfact 0.0 \
	--cs 2.7 \
	--defoc_low 0.8 \
	--defoc_high 3.0 \
	--noise_amp 0.050 \
	--noise_stdev 0.010 \
	--use_empirical_model True \
	--path_to_empirical_model /rugpfs/fs0/cem/store/achin/fsgs/training_network/generate_synth_data/empirical_pink_noise_model_k255e.mrcs \
	--lowpass_res 30 \
	--dil_rad 0 \
	--erod_rad 0  &
done


wait
