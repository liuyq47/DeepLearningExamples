#!/bin/bash
hosts=`cat $1`

path='/home/ec2-user/pt-bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT'
script='run_pretraining_32nodes_p1.sh'
count=0

for host in $hosts; do
    echo "$host"
    scp $path/scripts/$script $host:$path/scripts/
    ssh -t $host "cd $path && tmux new -d -s a && tmux send-keys -t a:0 'bash scripts/$script $count' Enter"
    count=$((count+1))
done
