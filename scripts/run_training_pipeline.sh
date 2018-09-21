#! /bin/sh
#
# run_training_pipeline.sh
# Copyright (C) 2018 rvedantam3 <vrama91@vt.edu>
#
# Distributed under terms of the MIT license.
#
# Script to run the CLEVR training pipeline.
source ~/venv/iep/bin/activate

PRIOR_CHECKPOINTS="/coc/scratch/rvedantam3/runs/pytorch_discovery/prior_local"

if [ ! -e ${PRIOR_CHECKPOINTS} ]; then
	mkdir ${PRIOR_CHECKPOINTS}
fi

# Training the program prior.
for lr in 0.001
do
	JOB_STRING='lr_'$lr
	RUN_TRAIN_DIR=${PRIOR_CHECKPOINTS}/${JOB_STRING}
	TRAINVAL_STRING="prior"

	if [ ! -e ${RUN_TRAIN_DIR} ]; then
		mkdir ${RUN_TRAIN_DIR}
	fi

  CMD_STRING="python scripts/train_model.py \
  	--model_type Prior \
		--load_train_features_memory=1 \
  	--num_iterations 20000 \
  	--checkpoint_every 1000 \
  	--learning_rate ${lr} \
  	--checkpoint_path ${RUN_TRAIN_DIR}/prior.pth"
	exec ${CMD_STRING}
  #source utils/invoke_slurm.sh "Y" "${CMD_STRING}" "${JOB_STRING}" "${TRAINVAL_STRING}" "${RUN_TRAIN_DIR}"
done


