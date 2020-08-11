#!/bin/bash
sleep 10.5h
hosts=`cat $1`

path='/home/ec2-user/pt-bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT'
script='run_pretraining_32_nodes_p1.4.sh'
count=0

for host in $hosts; do
    echo "$host"
    #scp $path/run_pretraining.py $host:$path/
    #scp $path/optimization.py $host:$path/
    #scp $path/modeling.py $host:$path/
    #scp $path/schedulers.py $host:$path/
    ssh -t $host "sudo pkill python"    
    ssh $host "mv $path/results/checkpoints/ckpt_3519.pt $path/results/checkpoints/ckpt_3519.pt.3"
    scp $path/scripts/$script $host:$path/scripts/
    ssh -t $host "cd $path && tmux new -d -s e && tmux send-keys -t e:0 'bash scripts/$script $count' Enter"
    count=$((count+1))
done
