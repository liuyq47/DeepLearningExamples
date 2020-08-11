#!/bin/bash
hosts=`cat $1`

path='/home/ec2-user/pt-bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT'
script='run_pretraining_192_nodes_p2.sh'
count=0

for host in $hosts; do
    echo "$host"
    #scp $path/run_pretraining.py $host:$path/
    #ssh -t $host "tmux kill-session -t a && sudo pkill python"
    #scp $path/results/checkpoints/ckpt_7038.pt $host:$path/results/checkpoints/
    #scp $path/run_pretraining.py $host:$path/
    #scp $path/modeling.py $host:$path/
    #ssh $host "mv $path/results/checkpoints/ckpt_3519.pt.2 $path/results/checkpoints/ckpt_3519.pt"
    scp $path/scripts/$script $host:$path/scripts/
    ssh -t $host "cd $path && tmux new -d -s b && tmux send-keys -t b:0 'bash scripts/$script $count' Enter"
    count=$((count+1))
done
