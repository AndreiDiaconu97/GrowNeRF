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

run_eval () {
	local path=${1}
    local checkpoint=${2}
	echo "Running eval..."
	echo "	path=${path}	checkpoint=${checkpoint}"

	CUDA_VISIBLE_DEVICES=${free_device} python eval_nerf.py --config ${path}/config.yml --checkpoint ${path}/${checkpoint} --savedir ${path}/rendered
}

run_train () {
	local cfg_path=${1}
	local run_name=${2}

	echo "Running train..."
	echo "	config=${cfg_path}"

	if [ -z "$run_name" ]; then local r=""; else local r="--run-name ${run_name}"; fi
	CUDA_VISIBLE_DEVICES=${free_device} python train_nerf.py --config ${cfg_path} $r
}

# generate_cache 
# run_eval logs/happy-darkness-112 checkpoint_stage02_epoch00000.ckpt

# run_train config/lego_baseline.yml baseline
# run_train config/lego_baseline_smaller.yml baseline_4l
# run_train config/lego_baseline_double_importance.yml baseline_128importance
# run_train config/lego_baseline_tanh.yml baseline_tanh
# run_train config/lego_baseline_kilonerf.yml baseline_kilonerf

# run_train config/lego_ensemble.yml ensemble_23000iters_2000corr_8weak_1layer
# run_train config/lego_ensemble.yml ensemble_23000iters_2000corr_8weak_1layer_tanh # CODE MODIFICATION
# run_train config/lego_ensemble_kilonerf_tanh.yml ensemble_kilonerf_noLRreset_tanh
# run_train config/lego_ensemble_kilonerf.yml ensemble_kilonerf_noLRreset
# run_train config/lego_ensemble.yml ensemble_23000iters_2000corr_8weak_1layer_corrLRnoReset_tanh # CODE MODIFICATION

# run_train config/lego_ensemble_kilonerf.yml ensemble_kilonerf_2000iters_8000corr
# run_train config/lego_ensemble_kilonerf.yml ensemble_kilonerf_2000iters_8000corr_tanh
# run_train config/lego_ensemble_kilonerf_fiftyfifty.yml ensemble_kilonerf_fiftyfifty
# run_train config/lego_ensemble_kilonerf.yml ensemble_kilonerf_resetEnsembleLR
# run_train config/lego_ensemble_kilonerf.yml ensemble_500iters_9500corr

run_train config/lego_ensemble_kilonerf.yml ensemble_4500iters_500corr_LRpeakDecay
run_train config/lego_ensemble_kilonerf_64.yml ensemble_kilonerf_4500iters_500corr_64_LRpeakDecay



# Last nohup PID: 645030

# TODO:
# don't reset training rates of weak or ensemble, or both
# try without fully corrective steps
# try with boosting rate
# TRY WITH NO VIEW-DEPENDENT PART
# TRY WITH CLASSICAL NERF (no flexibleNeRF)
# PUT skip connection every 4 (ORIGINAL) instead of 3