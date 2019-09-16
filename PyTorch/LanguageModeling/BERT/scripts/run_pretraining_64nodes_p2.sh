#!/bin/bash

# Specify dist training params
NUM_NODES=64
NODE_RANK=$1
MASTER_ADDR="172.31.11.20"
MASTER_PORT="1234"

# Specify phase 1 params
train_batch_size=128
learning_rate="6e-3"
precision="fp16"
num_gpus=8
warmup_proportion="0.2843"
train_steps=7038
save_checkpoint_steps=200
resume_training="false"
create_logfile="true"
accumulate_gradients="true"
gradient_accumulation_steps=2
seed=$RANDOM
job_name="bert_lamb_pretraining"
allreduce_post_accumulation="true"
allreduce_post_accumulation_fp16="true"
accumulate_into_fp16="false"

# Specify phase 1 data path
DATA_DIR=/home/ec2-user/pt-bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT/data/phase1/
BERT_CONFIG=bert_config.json
RESULTS_DIR=./results
CHECKPOINTS_DIR=./results/checkpoints

# Specify phase 2 params
train_batch_size_phase2=64
learning_rate_phase2="4e-3"
warmup_proportion_phase2="0.128"
train_steps_phase2=1563
gradient_accumulation_steps_phase2=8

#mkdir -p $CHECKPOINTS_DIR


if [ ! -d "$DATA_DIR" ] ; then
   echo "Warning! $DATA_DIR directory missing. Training cannot start"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be written to $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--resume_from_checkpoint"
fi

#Start Phase2

# Specify phase 2 data path
DATA_DIR=/home/ec2-user/pt-bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT/data/phase2/

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps_phase2"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

ACCUMULATE_INTO_FP16=""
if [ "$accumulate_into_fp16" == "true" ] ; then
   ACCUMULATE_INTO_FP16="--accumulate_into_fp16"
fi

echo $DATA_DIR
INPUT_DIR=$DATA_DIR
CMD=" /home/ec2-user/pt-bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size_phase2"
CMD+=" --max_seq_length=512"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --max_steps=$train_steps_phase2"
CMD+=" --warmup_proportion=$warmup_proportion_phase2"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate_phase2"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $ACCUMULATE_INTO_FP16"
CMD+=" --do_train --phase2 --resume_from_checkpoint --phase1_end_step=$train_steps"


CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus --nnodes=$NUM_NODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $CMD"


if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size_phase2 \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase2_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished phase2"
throughput=`cat $LOGFILE | grep Iteration | tail -1 | awk -F'it/s' '{print $1}' | awk -F',' '{print $2}' | egrep -o [0-9.]+`
loss=`cat $LOGFILE | grep 'Average Loss' | tail -1 | awk -F'Average Loss =' '{print $2}' | awk -F' ' '{print $1}' | egrep -o [0-9.]+`
final_loss=`cat $LOGFILE | grep 'Total Steps' | tail -1 | awk -F'Final Loss =' '{print $2}' | awk -F' ' '{print $1}' | egrep -o [0-9.]+`

train_perf=$(awk 'BEGIN {print ('$throughput' * '$num_gpus' * '$train_batch_size_phase2')}')
echo " training throughput phase2: $train_perf sequences/second"
echo "average loss: $loss"
echo "final loss: $final_loss"
