#!/bin/bash
hosts=`cat $1`

for host in $hosts; do
    echo "$host"
    ssh -t $host "tmux kill-session -t a && sudo pkill python"
done
