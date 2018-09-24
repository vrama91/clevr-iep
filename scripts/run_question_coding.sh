# Run question coding experiments for ICLR'19
source ~/venv/iep/bin/activate

#1] ROOT_DIR="/coc/scratch/rvedantam3/runs/pytorch_discovery/question_coding"
#2] ROOT_DIR="/coc/scratch/rvedantam3/runs/pytorch_discovery/question_coding_num_workers_1_load_train"
ROOT_DIR="/coc/scratch/rvedantam3/runs/pytorch_discovery/question_coding_no_eval_noimg"

PRIOR_CHECKPOINT="/coc/scratch/rvedantam3/runs/pytorch_discovery/prior/lr_0.001/prior.pth"
DEBUG=0

if [ ! -e ${ROOT_DIR} ]; then
  mkdir ${ROOT_DIR}
fi

# DEBUGGING.
if [ $DEBUG -eq "1" ]; then
  VERSION="iep"
  ROOT_DIR="/tmp/"
    JOB_STRING="$VERSION"
    RUN_TRAIN_DIR=${ROOT_DIR}/${JOB_STRING}
    TRAINVAL_STRING="question_coding"
  
    if [ ! -e ${RUN_TRAIN_DIR} ]; then
      mkdir ${RUN_TRAIN_DIR}
    fi
  
    CMD_STRING="python scripts/train_model.py \
      --model_type PG\
      --model_version $VERSION \
      --num_train_samples 200 \
      --program_prior_start_from ${PRIOR_CHECKPOINT} \
      --num_iterations 20000 \
      --checkpoint_every 500 \
      --mixing_factor_supervision 1.0 \
      --learning_rate 5e-4\
      --checkpoint_dir ${RUN_TRAIN_DIR}"
  
    exec ${CMD_STRING}
else
  # Run experiments with IEP.
  VERSION="iep"
  for supervision in 100 500 1000 2000 5000 10000 699989
  do
    JOB_STRING="$VERSION"_"${supervision}"
    RUN_TRAIN_DIR=${ROOT_DIR}/${JOB_STRING}
    TRAINVAL_STRING="question_coding"
  
    if [ ! -e ${RUN_TRAIN_DIR} ]; then
      mkdir ${RUN_TRAIN_DIR}
    fi

    if [ $supervision -lt 5000 ]; then
      CHECKPOINT_EVERY=100
      NUM_ITERATIONS=5000
    else
      CHECKPOINT_EVERY=2000
      NUM_ITERATIONS=20000
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
      --learning_rate 1e-3\
      --checkpoint_dir ${RUN_TRAIN_DIR}"

    #if [ $supervision -gt "1000" ]; then
    #  CMD_STRING=${CMD_STRING}" --load_train_features_memory"
    #fi
    # exec ${CMD_STRING}
    source utils/invoke_slurm.sh "Y" "${CMD_STRING}" "${JOB_STRING}" "${TRAINVAL_STRING}" "${RUN_TRAIN_DIR}"
  done
  
  # Run experiments with discovery.
  VERSION="discovery"

  SSL_ALPHA=100.0
  KL_BETA=0.1
  MIXING_FACTOR=1.0

  for supervision in 100 500 1000 2000 5000 10000 699989
  do
    JOB_STRING="$VERSION"_"${supervision}"_mix_"$MIXING_FACTOR"_ssl_"${SSL_ALPHA}"_kl_"$KL_BETA"
    RUN_TRAIN_DIR=${ROOT_DIR}/${JOB_STRING}
    TRAINVAL_STRING="question_coding"
  
    if [ ! -e ${RUN_TRAIN_DIR} ]; then
      mkdir ${RUN_TRAIN_DIR}
    fi

    if [ $supervision -lt 5000 ]; then
      CHECKPOINT_EVERY=100
      NUM_ITERATIONS=5000
    else
      CHECKPOINT_EVERY=2000
      NUM_ITERATIONS=20000
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
      --num_iterations 20000 \
      --checkpoint_every ${CHECKPOINT_EVERY} \
      --mixing_factor_supervision ${MIXING_FACTOR} \
      --learning_rate 1e-3\
      --checkpoint_dir ${RUN_TRAIN_DIR}"

    #if [ $supervision -gt "100" ]; then
    #  CMD_STRING=${CMD_STRING}" --load_train_features_memory"
    #fi
  
    #exec ${CMD_STRING}
    source utils/invoke_slurm.sh "Y" "${CMD_STRING}" "${JOB_STRING}" "${TRAINVAL_STRING}" "${RUN_TRAIN_DIR}"
  done

  # MORE DETAILED RUNS FOR 500
  for supervision in 500 1000 2000
  do
    for SSL_ALPHA in 10.0 200.0 1000.0
    do
      JOB_STRING="$VERSION"_"${supervision}"_mix_"$MIXING_FACTOR"_ssl_"${SSL_ALPHA}"_kl_"$KL_BETA"
      RUN_TRAIN_DIR=${ROOT_DIR}/${JOB_STRING}
      TRAINVAL_STRING="question_coding"
  
      if [ ! -e ${RUN_TRAIN_DIR} ]; then
        mkdir ${RUN_TRAIN_DIR}
      fi

      if [ $supervision -lt 1000 ]; then
        CHECKPOINT_EVERY=100
      else
        CHECKPOINT_EVERY=1000
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
        --num_iterations 20000 \
        --checkpoint_every ${CHECKPOINT_EVERY} \
        --mixing_factor_supervision ${MIXING_FACTOR} \
        --learning_rate 1e-3\
        --checkpoint_dir ${RUN_TRAIN_DIR}"

      #if [ $supervision -gt "100" ]; then
      #  CMD_STRING=${CMD_STRING}" --load_train_features_memory"
      #fi
  
      #exec ${CMD_STRING}
      source utils/invoke_slurm.sh "Y" "${CMD_STRING}" "${JOB_STRING}" "${TRAINVAL_STRING}" "${RUN_TRAIN_DIR}"
    done
  done
fi