#!/bin/bash
hosts=`cat $1`

path='/home/ec2-user/pt-bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT'
script='run_pretraining_64nodes_p1.sh'
count=0

for host in $hosts; do
    echo "$host"
    ssh -t $host "tmux new -d -s a && tmux send-keys -t a:0 'curl -O https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-1.5.4.tar.gz && tar -xf aws-efa-installer-1.5.4.tar.gz && cd aws-efa-installer && sudo ./efa_installer.sh -y' Enter"    
#scp $path/scripts/$script $host:$path/scripts/
    #ssh -t $host "mkdir /home/ec2-user/pt-bert/NV-data/seq_128_pred_20_dupe_5/training"
#ssh -t $host "rm -r /home/ec2-user/pt-bert/NV-data/training && mkdir -p /home/ec2-user/pt-bert/NV-data/seq_128_pred_20_dupe_5/training && mv /home/ec2-user/pt-bert/NV-data/test /home/ec2-user/pt-bert/NV-data/seq_128_pred_20_dupe_5/"
    #ssh -t $host "aws s3 cp s3://shaabhn-bert/NV-data/training/ /home/ec2-user/pt-bert/NV-data/seq_128_pred_20_dupe_5/training/ --recursive" 
    #ssh -t $host "cd $path && tmux new -d -s a && tmux send-keys -t a:0 'bash scripts/$script $count' Enter"
    #count=$((count+1))
done

#for host in $hosts; do
 #   echo "$host"
 #   ssh -t $host "cd $path && tmux new -d -s a && tmux send-keys -t a:0 'bash scripts/$script $count' Enter"
 #   count=$((count+1))
#done
