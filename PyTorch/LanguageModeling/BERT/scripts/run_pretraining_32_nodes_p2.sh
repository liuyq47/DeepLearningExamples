#!/bin/bash

export LD_LIBRARY_PATH=$HOME/aws-ofi-nccl/install/lib/:$HOME/nccl/build/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/usr/local/mpi/lib:$LD_LIBRARY_PATH
export FI_PROVIDER="efa"
export FI_EFA_TX_MIN_CREDITS=64
#export NCCL_TREE_THRESHOLD=15360000
export NCCL_TREE_THRESHOLD=16000000000
export NCCL_MIN_NRINGS=1
export NCCL_DEBUG=VERSION
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=eth0

export FI_OFI_RXR_RX_COPY_UNEXP=1
export FI_OFI_RXR_RX_COPY_OOO=1
export FI_EFA_MR_CACHE_ENABLE=1
export FI_OFI_RXR_INLINE_MR_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Specify dist training params
NUM_NODES=32
NODE_RANK=$1
MASTER_ADDR="172.31.39.57"
MASTER_PORT="1234"

# Specify phase 1 params
train_batch_size=384
learning_rate="0.00675"
precision="fp16"
num_gpus=8
optimizer="neslamb"
warmup_proportion="0.4265"
const_proportion="0.2735"
train_steps=3519
save_checkpoint_steps=8000
resume_training="false"
create_logfile="true"
accumulate_gradients="true"
gradient_accumulation_steps=6
seed=$RANDOM
job_name="bert_neslamb_pretraining"
allreduce_post_accumulation="true"
allreduce_post_accumulation_fp16="true"
accumulate_into_fp16="false"

# Specify phase 1 data path
#DATA_DIR=/fsx/datasets/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_256/books_wiki_en_corpus_train/
#DATA_DIR=/home/ec2-user/bert_data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_10_shard_1536_nv/books_wiki_en_corpus_train
#DATA_DIR=/home/ec2-user/bert_data_pytorch/bert_data_2048/phase1/training
DATA_DIR=/home/ec2-user/bert_data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_10_shard_1536_small/books_wiki_en_corpus_train
BERT_CONFIG=./bert_config.json
RESULTS_DIR=./results
CHECKPOINTS_DIR=./results/checkpoints

# Specify phase 2 params
train_batch_size_phase2=132
#train_batch_size_phase2=192
learning_rate_phase2="5e-3"
warmup_proportion_phase2="0.192"
const_proportion_phase2="0.108"
train_steps_phase2=782
gradient_accumulation_steps_phase2=12
#gradient_accumulation_steps_phase2=16

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
#DATA_DIR=/fsx/datasets/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_256/books_wiki_en_corpus_train/
#DATA_DIR=/fsx/datasets/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1536_small/books_wiki_en_corpus_train
DATA_DIR=/home/ec2-user/bert_data/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_10_shard_1536_small/books_wiki_en_corpus_train
#DATA_DIR=/home/ec2-user/bert_data/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_10_shard_1536_nv/books_wiki_en_corpus_train
#DATA_DIR=/home/ec2-user/bert_data_pytorch/bert_data_2048/phase2/training
#DATA_DIR=/home/ec2-user/bert_data_pytorch/bert_data_2048_wwm/phase2

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
CMD+=" --optimizer=$optimizer"
CMD+=" --warmup_proportion=$warmup_proportion_phase2"
CMD+=" --const_proportion=$const_proportion_phase2"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate_phase2"
CMD+=" --resume_step=$train_steps"
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
