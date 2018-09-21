#! /bin/sh
#
# preprocess_${PROCESSED_FEATURES}.sh
# Copyright (C) 2018 rvedantam3 <vrama91@vt.edu>
#
# Distributed under terms of the MIT license.
#
CLEVR_DATASET="/srv/share/datasets/clevr/CLEVR_v1.0"
PROCESSED_FEATURES=${CLEVR_DATASET}/clevr-iep-data

python scripts/extract_features.py \
	  --input_image_dir ${CLEVR_DATASET}/images/train \
	  --output_h5_file ${PROCESSED_FEATURES}/train_features.h5

python scripts/extract_features.py \
	  --input_image_dir ${CLEVR_DATASET}/images/val \
	  --output_h5_file ${PROCESSED_FEATURES}/val_features.h5

python scripts/extract_features.py \
	  --input_image_dir ${CLEVR_DATASET}/images/test \
	  --output_h5_file ${PROCESSED_FEATURES}/test_features.h5

python scripts/preprocess_questions.py \
	  --input_questions_json ${CLEVR_DATASET}/questions/CLEVR_train_questions.json \
	  --output_h5_file ${PROCESSED_FEATURES}/train_questions.h5 \
	  --output_vocab_json ${PROCESSED_FEATURES}/vocab.json

python scripts/preprocess_questions.py \
	  --input_questions_json ${CLEVR_DATASET}/questions/CLEVR_val_questions.json \
	  --output_h5_file ${PROCESSED_FEATURES}/val_questions.h5 \
	  --input_vocab_json ${PROCESSED_FEATURES}/vocab.json
  
python scripts/preprocess_questions.py \
	  --input_questions_json ${CLEVR_DATASET}/questions/CLEVR_test_questions.json \
	  --output_h5_file ${PROCESSED_FEATURES}/test_questions.h5 \
	  --input_vocab_json ${PROCESSED_FEATURES}/vocab.json

