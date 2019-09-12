#!/bin/bash
hosts=`cat $1`

path='/home/ec2-user/pt-bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT'
script='run_pretraining_32nodes_p2.sh'
count=0

for host in $hosts; do
    echo "$host"
    scp $path/scripts/$script $host:$path/scripts/
    scp $path/results/checkpoints/ckpt_7038.pt $host:$path/results/checkpoints/
    ssh -t $host "tmux kill-session -t a && cd $path && tmux new -d -s a && tmux send-keys -t a:0 'bash scripts/$script $count' Enter"
    count=$((count+1))
done
