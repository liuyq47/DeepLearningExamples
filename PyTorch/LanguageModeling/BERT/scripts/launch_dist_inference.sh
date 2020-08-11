#!/bin/bash

path='/home/ec2-user/pt-bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT'
script='run_pretraining_inference.sh'
count=0

tmux kill-session -t a && sudo pkill python
cd $path && tmux new -d -s a && tmux send-keys -t a:0 "bash scripts/$script 64 fp16 8 eval 8601" Enter
