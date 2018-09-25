#! /bin/sh
#
# invoke_slurm.sh
# Copyright (C) 2018 rvedantam3 <vrama91@vt.edu>
#
# Distributed under terms of the MIT license.
#
# Script to invoke slurm and run a job if there are enough GPUs available.
GPU_THRESHOLD=5
SLEEP=600
GPU_JOB=${1} # Y or N
CMD_STRING=${2} 
RUN_JOB_STRING=${3}
TRAINVAL_STRING=${4}
RUN_TRAIN_DIR=${5}

function launch_jobs() {
	echo "GPU available, launching jobs."
	# CPU Jobs also get launched only when GPUs are available currently.
	if [ ${GPU_JOB} = "Y" ]; then
		echo "Launching GPU job $RUN_JOB_STRING"
		sbatch utils/slurm_wrapper.sh "${CMD_STRING}" "${RUN_JOB_STRING}" "${TRAINVAL_STRING}" "${RUN_TRAIN_DIR}" > /tmp/log
	else
		echo "Launching CPU job $RUN_JOB_STRING"
		sbatch utils/slurm_wrapper_cpu_only.sh "${CMD_STRING}" "${RUN_JOB_STRING}" "${TRAINVAL_STRING}" "${RUN_TRAIN_DIR}" > /tmp/log
	fi
}

_PID=$$
SCHEDULED_FILE="/tmp/scheduled_"${_PID}
if [ -e ${SCHEDULED_FILE} ]; then
	rm ${SCHEDULED_FILE}
fi

while [ ! -e ${SCHEDULED_FILE} ]; 
do
	# Wait for GPUs
	#while [ `gpus_free` -lt $GPU_THRESHOLD ]; 
	#do
  #		echo "Sleeping for $SLEEP"
#		date | awk '{print "Waiting for GPUs: " $1" "$2" "$3" "$4}'
#		sleep ${SLEEP}
#	done
#	(
#	flock -x 200
#	if ! [ `gpus_free` -lt $GPU_THRESHOLD ]; then
		launch_jobs
		# We are using the creation of a file to check that the process was launched
		# because there does not seem to be an easy way to set variables in this
		# flock scope and have it reflect in the outer scope.
		touch ${SCHEDULED_FILE}
#	fi
#	) 200>/var/lock/gpu_process_lock_outer
done

rm ${SCHEDULED_FILE}
