# Denser sampling of supervision.
source ~/venv/iep/bin/activate

#1] ROOT_DIR="/coc/scratch/rvedantam3/runs/pytorch_discovery/question_coding"
#2] ROOT_DIR="/coc/scratch/rvedantam3/runs/pytorch_discovery/question_coding_num_workers_1_load_train"
ROOT_DIR="/coc/scratch/rvedantam3/runs/pytorch_discovery/question_coding_denser"

PRIOR_CHECKPOINT="/coc/scratch/rvedantam3/runs/pytorch_discovery/prior/lr_0.001/prior.pth"
LEARNING_RATE=1e-3

# Whether we are training or evaluating.
DEBUG=0
TRAIN=1

VAL_SPLIT="train"
NUM_VAL_SAMPLES=5000

if [ ! -e ${ROOT_DIR} ]; then
  mkdir ${ROOT_DIR}
fi

# Run experiments with IEP.
VERSION="iep"
for supervision in 400 500 600 700 800 900 1000
do
  CHECKPOINT_EVERY=500
  NUM_ITERATIONS=20000
  JOB_STRING="$VERSION"_"${supervision}"
  RUN_TRAIN_DIR=${ROOT_DIR}/${JOB_STRING}
  TRAINVAL_STRING="question_coding"

  if [ ! -e ${RUN_TRAIN_DIR} ]; then
    mkdir ${RUN_TRAIN_DIR}
  fi

  CMD_STRING="python scripts/train_model.py \
    --model_type PG\
    --loader_num_workers 1\
    --dont_load_train_features_memory\
    --model_version $VERSION \
    --program_prior_start_from ${PRIOR_CHECKPOINT} \
    --program_supervision_npy /srv/share/datasets/clevr/CLEVR_v1.0/clevr-iep-data/semi_supervised_train_$supervision.npy \
    --num_iterations ${NUM_ITERATIONS} \
    --checkpoint_every ${CHECKPOINT_EVERY} \
    --mixing_factor_supervision 1.0 \
    --learning_rate ${LEARNING_RATE}\
    --checkpoint_dir ${RUN_TRAIN_DIR}"
  
  #if [ $supervision -gt "1000" ]; then
  #  CMD_STRING=${CMD_STRING}" --load_train_features_memory"
  #fi

  if [ ${TRAIN} -eq "0" ]; then
    CMD_STRING=${CMD_STRING}" --only_evaluation_split ${VAL_SPLIT} --num_val_samples ${NUM_VAL_SAMPLES}\
      --val_question_h5 /srv/share/datasets/clevr/CLEVR_v1.0/clevr-iep-data/${VAL_SPLIT}_questions.h5\
      --val_features_h5 /srv/share/datasets/clevr/CLEVR_v1.0/clevr-iep-data/${VAL_SPLIT}_features.h5"
    echo ${CMD_STRING}
  else
    source utils/invoke_slurm.sh "Y" "${CMD_STRING}" "${JOB_STRING}" "${TRAINVAL_STRING}" "${RUN_TRAIN_DIR}"
  fi

done

# Run experiments with discovery.
VERSION="discovery"

SSL_ALPHA=100.0
KL_BETA=0.1
MIXING_FACTOR=1.0
CHECKPOINT_EVERY=500
NUM_ITERATIONS=20000

for supervision in 400 500 600 700 800 900 1000
do
  JOB_STRING="$VERSION"_"${supervision}"_mix_"$MIXING_FACTOR"_ssl_"${SSL_ALPHA}"_kl_"$KL_BETA"
  RUN_TRAIN_DIR=${ROOT_DIR}/${JOB_STRING}
  TRAINVAL_STRING="question_coding"

  if [ ! -e ${RUN_TRAIN_DIR} ]; then
    mkdir ${RUN_TRAIN_DIR}
  fi

  CMD_STRING="python scripts/train_model.py \
    --model_type PG\
    --loader_num_workers 1\
    --dont_load_train_features_memory\
    --model_version $VERSION \
    --program_prior_start_from ${PRIOR_CHECKPOINT} \
    --discovery_alpha ${SSL_ALPHA} \
    --discovery_beta ${KL_BETA} \
    --program_supervision_npy /srv/share/datasets/clevr/CLEVR_v1.0/clevr-iep-data/semi_supervised_train_$supervision.npy \
    --num_iterations ${NUM_ITERATIONS} \
    --checkpoint_every ${CHECKPOINT_EVERY} \
    --mixing_factor_supervision ${MIXING_FACTOR} \
    --learning_rate ${LEARNING_RATE}\
    --checkpoint_dir ${RUN_TRAIN_DIR}"
  
  #if [ $supervision -gt "1000" ]; then
  #  CMD_STRING=${CMD_STRING}" --load_train_features_memory"
  #fi

  if [ ${TRAIN} -eq "0" ]; then
    CMD_STRING=${CMD_STRING}" --only_evaluation_split ${VAL_SPLIT} --num_val_samples ${NUM_VAL_SAMPLES} \
      --val_question_h5 /srv/share/datasets/clevr/CLEVR_v1.0/clevr-iep-data/${VAL_SPLIT}_questions.h5\
      --val_features_h5 /srv/share/datasets/clevr/CLEVR_v1.0/clevr-iep-data/${VAL_SPLIT}_features.h5"
    echo ${CMD_STRING}
  else
    source utils/invoke_slurm.sh "Y" "${CMD_STRING}" "${JOB_STRING}" "${TRAINVAL_STRING}" "${RUN_TRAIN_DIR}"
  fi
done

### MORE DETAILED RUNS FOR 500, 100, 2000 changing SSL_ALPHA
for supervision in 400 500 600 700 800 900 1000
do
  for SSL_ALPHA in 10.0 200.0 1000.0
  do
    JOB_STRING="$VERSION"_"${supervision}"_mix_"$MIXING_FACTOR"_ssl_"${SSL_ALPHA}"_kl_"$KL_BETA"
    RUN_TRAIN_DIR=${ROOT_DIR}/${JOB_STRING}
    TRAINVAL_STRING="question_coding"

    if [ ! -e ${RUN_TRAIN_DIR} ]; then
      mkdir ${RUN_TRAIN_DIR}
    fi

    CMD_STRING="python scripts/train_model.py \
      --model_type PG\
      --dont_load_train_features_memory\
      --loader_num_workers 1\
      --model_version $VERSION \
      --program_prior_start_from ${PRIOR_CHECKPOINT} \
      --discovery_alpha ${SSL_ALPHA} \
      --discovery_beta ${KL_BETA} \
      --program_supervision_npy /srv/share/datasets/clevr/CLEVR_v1.0/clevr-iep-data/semi_supervised_train_$supervision.npy \
      --num_iterations ${NUM_ITERATIONS} \
      --checkpoint_every ${CHECKPOINT_EVERY} \
      --mixing_factor_supervision ${MIXING_FACTOR} \
      --learning_rate ${LEARNING_RATE}\
      --checkpoint_dir ${RUN_TRAIN_DIR}"

    #if [ $supervision -gt "1000" ]; then
    #  CMD_STRING=${CMD_STRING}" --load_train_features_memory"
    #fi

    if [ ${TRAIN} -eq "0" ]; then
      CMD_STRING=${CMD_STRING}" --only_evaluation_split ${VAL_SPLIT}  --num_val_samples ${NUM_VAL_SAMPLES}\
      --val_question_h5 /srv/share/datasets/clevr/CLEVR_v1.0/clevr-iep-data/${VAL_SPLIT}_questions.h5\
      --val_features_h5 /srv/share/datasets/clevr/CLEVR_v1.0/clevr-iep-data/${VAL_SPLIT}_features.h5"
      echo ${CMD_STRING}
    else
      source utils/invoke_slurm.sh "Y" "${CMD_STRING}" "${JOB_STRING}" "${TRAINVAL_STRING}" "${RUN_TRAIN_DIR}"
    fi
  done
done
 
## MORE DETAILED RUNS FOR 500, 100, 2000 changing KL_BETA
SSL_ALPHA=100.0
for supervision in 400 500 600 700 800 900 1000
do
  for KL_BETA in 0.3 0.5 0.7 
  do
    JOB_STRING="$VERSION"_"${supervision}"_mix_"$MIXING_FACTOR"_ssl_"${SSL_ALPHA}"_kl_"$KL_BETA"
    RUN_TRAIN_DIR=${ROOT_DIR}/${JOB_STRING}
    TRAINVAL_STRING="question_coding"

    if [ ! -e ${RUN_TRAIN_DIR} ]; then
      mkdir ${RUN_TRAIN_DIR}
    fi

    CMD_STRING="python scripts/train_model.py \
      --model_type PG\
      --dont_load_train_features_memory\
      --loader_num_workers 1\
      --model_version $VERSION \
      --program_prior_start_from ${PRIOR_CHECKPOINT} \
      --discovery_alpha ${SSL_ALPHA} \
      --discovery_beta ${KL_BETA} \
      --program_supervision_npy /srv/share/datasets/clevr/CLEVR_v1.0/clevr-iep-data/semi_supervised_train_$supervision.npy \
      --num_iterations ${NUM_ITERATIONS} \
      --checkpoint_every ${CHECKPOINT_EVERY} \
      --mixing_factor_supervision ${MIXING_FACTOR} \
      --learning_rate ${LEARNING_RATE}\
      --checkpoint_dir ${RUN_TRAIN_DIR}"

    #if [ $supervision -gt "1000" ]; then
    #  CMD_STRING=${CMD_STRING}" --load_train_features_memory"
    #fi

    if [ ${TRAIN} -eq "0" ]; then
      CMD_STRING=${CMD_STRING}" --only_evaluation_split ${VAL_SPLIT} --num_val_samples ${NUM_VAL_SAMPLES}\
      --val_question_h5 /srv/share/datasets/clevr/CLEVR_v1.0/clevr-iep-data/${VAL_SPLIT}_questions.h5\
      --val_features_h5 /srv/share/datasets/clevr/CLEVR_v1.0/clevr-iep-data/${VAL_SPLIT}_features.h5"
      echo ${CMD_STRING}
    else
      source utils/invoke_slurm.sh "Y" "${CMD_STRING}" "${JOB_STRING}" "${TRAINVAL_STRING}" "${RUN_TRAIN_DIR}"
    fi
  done
done