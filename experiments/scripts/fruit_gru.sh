#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET_NAME=ResidualGRUNet
EXP_DETAIL=fruit
DATASET='./experiments/dataset/'$EXP_DETAIL'.json'
OUT_PATH='./output/'$NET_NAME/$EXP_DETAIL
LOG="$OUT_PATH/log.`date +'%Y-%m-%d_%H-%M-%S'`"

# Make the dir if it not there
mkdir -p $OUT_PATH
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export THEANO_FLAGS="floatX=float32,device=gpu,assert_no_cpu_op='raise'"

python main.py \
      --batch-size 2 \
      --iter 5000 \
      --cfg experiments/cfgs/no_crop.yaml \
      --dataset $DATASET \
      --out $OUT_PATH \
      --model $NET_NAME \
      ${*:1}
