#!/bin/bash
hosts=`cat $1`


for host in $hosts; do
    ssh -A $host 'cd ~/apex/ && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_multihead_attn" ./ --user' &
done     
