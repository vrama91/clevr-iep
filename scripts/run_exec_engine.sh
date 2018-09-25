# Run execution engine experiments for ICLR'19.
source ~/venv/iep/bin/activate

ROOT_DIR="/coc/scratch/rvedantam3/runs/pytorch_discovery/execution_engine"
QUESTION_CODING_BASE_DIR="/coc/scratch/rvedantam3/runs/pytorch_discovery/question_coding_no_eval_noimg"

TRAIN=1
LEARNING_RATE=5e-4

if [ ! -e ${ROOT_DIR} ]; then
  mkdir ${ROOT_DIR}
fi

get_qc_checkpoint(){
    if [ ! -e $QUESTION_CODING_BASE_DIR/$1/trainbest_checkpoint.json ]; then
      echo "NOT FOUND!!"
      return
    else
      echo `cat $QUESTION_CODING_BASE_DIR/$1/trainbest_checkpoint.json | tr "\"" " " | tr " " "\n" | grep "ckpt" | sed "s/}//g"`
    fi
}

VERSION="iep"
NUM_ITERATIONS=40000
#CHECKPOINT_EVERY=4000 # TODO(vrama): change
CHECKPOINT_EVERY=100 # TODO(vrama): change

#for supervision in 699989 100 500 1000 2000 5000 10000 699989
for supervision in 699989 
do
  JOB_STRING="$VERSION"_"${supervision}"
  RUN_TRAIN_DIR=${ROOT_DIR}/${JOB_STRING}
  TRAINVAL_STRING="exec_engine"

  BEST_PG_CHECKPOINT=`get_qc_checkpoint ${JOB_STRING}`
  echo "Using question coding checkpoint: "$BEST_PG_CHECKPOINT

  if [ ! -e ${RUN_TRAIN_DIR} ]; then
    mkdir ${RUN_TRAIN_DIR}
  fi

  CMD_STRING="python scripts/train_model.py \
    --model_type EE\
    --model_version $VERSION \
    --preload_image_features_ram\
    --loader_num_workers 1\
    --program_generator_start_from ${BEST_PG_CHECKPOINT}\
    --num_iterations ${NUM_ITERATIONS} \
    --evaluate_during_train \
    --checkpoint_every ${CHECKPOINT_EVERY} \
    --learning_rate ${LEARNING_RATE}\
    --checkpoint_dir ${RUN_TRAIN_DIR}"

  if [ ${TRAIN} -eq 0 ]; then
    CMD_STRING=$CMD_STRING' --only_evaluation_split val'
  else
    exec $CMD_STRING
    #source utils/invoke_slurm.sh "Y" "${CMD_STRING}" "${JOB_STRING}" "${TRAINVAL_STRING}" "${RUN_TRAIN_DIR}"
  fi
done