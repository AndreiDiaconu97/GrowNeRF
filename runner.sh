#!/bin/bash

# get the most free GPU (in terms of memory)
# free_device=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs)
free_device=1
echo "Most free GPU: ${free_device}"


generate_cache () {
	echo "Generating cache..."
	python cache_dataset.py --datapath cache/nerf_synthetic/lego/ --savedir cache/legocache/legofull --num-random-rays 8192 --num-variations 50 --type blender
	echo "Cache created"
}

run_lego="CUDA_VISIBLE_DEVICES=${free_device} python train_nerf.py --config config/grownet_lego.yml --max_mins 180"

# MISC #
# eval $run_lego --run-name ensemble_default_newDecay
# eval $run_lego --run-name kilonerf_baseline_500000 				--depth 3 --width 32 --weak_iters 500000 --corrective_iters 0  --n_stages 1
# eval $run_lego --run-name kilonerf_baseline_500000_longerDecay 	--depth 3 --width 32 --weak_iters 500000 --corrective_iters 0  --n_stages 1 --lr_decay_weak 500000
# eval $run_lego --run-name kilonerf64_baseline_500000 				--depth 3 --width 64 --weak_iters 500000 --corrective_iters 0  --n_stages 1

# TEST TRAIN/CORRECTIVE RATIO #
# eval $run_lego --run-name kilonerf_D3W32_9000WI1000CI_N20 	--depth 3 --width 32 --weak_iters 9000 --corrective_iters 1000 --n_stages 20

# eval $run_lego --run-name kilonerf_D3W32_4500WI500CI_N40 	--depth 3 --width 32 --weak_iters 4500 --corrective_iters 500  --n_stages 40
# eval $run_lego --run-name kilonerf_D3W32_5000WI0CI_N40 		--depth 3 --width 32 --weak_iters 5000 --corrective_iters 0    --n_stages 40
# eval $run_lego --run-name kilonerf_D3W32_5000WI0CI_N40_tanh --depth 3 --width 32 --weak_iters 5000 --corrective_iters 0    --n_stages 40 --render_activation_fn tanh
# eval $run_lego --run-name kilonerf_D3W32_500WI4500CI_N40 	--depth 3 --width 32 --weak_iters 500  --corrective_iters 4500 --n_stages 40
# eval $run_lego --run-name kilonerf_D3W32_1WI5000CI_N40 		--depth 3 --width 32 --weak_iters 1    --corrective_iters 5000 --n_stages 40

# TEST LEARNING RATE DECAY #
# eval $run_lego --run-name kilonerf_D3W32_4500WI500CI_N40_decayW4500 		  --depth 3 --width 32 --weak_iters 4500 --corrective_iters 500 --n_stages 40 --lr_decay_weak 4500
# eval $run_lego --run-name kilonerf_D3W32_4500WI500CI_N40_noRSTweak 		      --depth 3 --width 32 --weak_iters 4500 --corrective_iters 500 --n_stages 40 --lr_reset_weak False
# eval $run_lego --run-name kilonerf_D3W32_4500WI500CI_N40_decayC500_RSTcorr    --depth 3 --width 32 --weak_iters 4500 --corrective_iters 500 --n_stages 40 --lr_decay_corrective 500 --lr_reset_corrective True
# eval $run_lego --run-name kilonerf_D3W32_500WI4500CI_N40_decayC100000_Peak0.5 --depth 3 --width 32 --weak_iters 500 --corrective_iters 4500 --n_stages 40 --lr_decay_corrective 100000 --lr_decay_corrective_peaked 0.5

# TEST NETWORK SIZES #
# eval $run_lego --run-name kilonerf_D2W32_4500WI500CI_N40   --depth 2 --width 32 --weak_iters 4500 --corrective_iters 500   --n_stages 40
# eval $run_lego --run-name kilonerf_D3W32_4500WI500CI_N40   --depth 3 --width 32 --weak_iters 4500 --corrective_iters 500   --n_stages 40
# eval $run_lego --run-name kilonerf_D4W32_4500WI500CI_N40   --depth 4 --width 32 --weak_iters 4500 --corrective_iters 500   --n_stages 40

eval $run_lego --run-name kilonerf_D2W64_2500WI2500CI_N40   --depth 2 --width 64 --weak_iters 2500 --corrective_iters 2500  --n_stages 40
eval $run_lego --run-name kilonerf_D3W64_2500WI2500CI_N40   --depth 3 --width 64 --weak_iters 2500 --corrective_iters 2500  --n_stages 40
eval $run_lego --run-name kilonerf_D4W64_2500WI2500CI_N40   --depth 4 --width 64 --weak_iters 2500 --corrective_iters 2500  --n_stages 40

# OTHER #
# eval $run_lego --run-name ensemble_D2W128_N8 			   --depth 2 													 --n_stages 8
# eval $run_lego --run-name ensemble_D2W128_9000WI1000CI_N16 --depth 2 		   --weak_iters 9000 --corrective_iters 1000 --n_stages 16
# eval $run_lego --run-name ensemble_D2W128_N8_tanh			   		--depth 2 											 --n_stages 8 	--render_activation_fn tanh
# eval $run_lego --run-name ensemble_D2W128_9000WI1000CI_N16_tanh 	--depth 2  --weak_iters 9000 --corrective_iters 1000 --n_stages 16 	--render_activation_fn tanh
# eval $run_lego --run-name ensemble_D2W128_1000WI9000CI_N16 			--depth 2  --weak_iters 1000 --corrective_iters 9000 --n_stages 16
# eval $run_lego --run-name ensemble_D2W128_500WI4500CI_N16 			--depth 2  --weak_iters 500  --corrective_iters 4500 --n_stages 32
eval $run_lego --run-name kilonerf_D3W64_10000WI10000CI_N40   --depth 3 --width 64 --weak_iters 10000 --corrective_iters 10000  --n_stages 40

# TEST LEARNING RATE VALUES
# eval $run_lego --run-name kilonerf_D3W32_4500WI500CI_N40_LRhigh 	--depth 3 --width 32 --weak_iters 4500 --corrective_iters 500  --n_stages 40 --lr 5.0E-2
# eval $run_lego --run-name kilonerf_D3W32_4500WI500CI_N40_LRlow 		--depth 3 --width 32 --weak_iters 4500 --corrective_iters 500  --n_stages 40 --lr 5.0E-4
# eval $run_lego --run-name kilonerf_D3W32_500WI4500CI_N40_ensLRhigh 	--depth 3 --width 32 --weak_iters 500  --corrective_iters 4500 --n_stages 40 --lr_ensemble 5.0E-2
# eval $run_lego --run-name kilonerf_D3W32_500WI4500CI_N40_ensLRlow 	--depth 3 --width 32 --weak_iters 500  --corrective_iters 4500 --n_stages 40 --lr_ensemble 5.0E-4


# TEST BOOSTING RATE
eval $run_lego --run-name kilonerf_D3W64_2500WI2500CI_N40   --depth 3 --width 64 --weak_iters 2500 --corrective_iters 2500  --n_stages 40 --learn_boost_rate True

# TEST HIERARCICHAL ENSEMBLE
eval $run_lego --run-name kilonerf_D3W256_2500WI2500CI_H0.8_N40   --depth 3 --width 256 --weak_iters 2500 --corrective_iters 2500  --n_stages 40 --hierarchical_factor 0.8
eval $run_lego --run-name kilonerf_D3W32_2500WI2500CI_H1.2_N40   --depth 3 --width 32 --weak_iters 2500 --corrective_iters 2500  --n_stages 40 --hierarchical_factor 1.2

eval $run_lego --run-name kilonerf_D3W256_5000WI5000CI_H0.8_N40   --depth 3 --width 256 --weak_iters 5000 --corrective_iters 5000  --n_stages 40 --hierarchical_factor 0.8
eval $run_lego --run-name kilonerf_D3W32_5000WI5000CI_H1.2_N40   --depth 3 --width 32 --weak_iters 5000 --corrective_iters 5000  --n_stages 40 --hierarchical_factor 1.2

eval $run_lego --run-name kilonerf_D3W32_5000WI5000CI_H1.5_N40   --depth 3 --width 32 --weak_iters 5000 --corrective_iters 5000  --n_stages 40 --hierarchical_factor 1.5


# Last nohup PID: 163590

# TODO:
# don't reset training rates of weak or ensemble, or both
# TRY WITH NO VIEW-DEPENDENT PART
# TRY WITH CLASSICAL NERF (no flexibleNeRF)
# TRY SINGLE COARSE + FINE ENSEMBLE
# TRY WITHOUT CONTEXT PROPAGATION

# MULTIRESOLUTION
# - FROM BIG TO SMALL <- THIS
# - FROM SMALL TO BIG
