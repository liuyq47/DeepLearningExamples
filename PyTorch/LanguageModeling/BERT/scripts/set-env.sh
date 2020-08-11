#!/bin/bash
hosts=`cat $1`

for host in $hosts; do
    echo "$host"
    #ssh -t $host 'export FI_PROVIDER="efa" && export FI_EFA_TX_MIN_CREDITS=64 && export NCCL_TREE_THRESHOLD=15360000 && export NCCL_MIN_NRINGS=1 && export NCCL_DEBUG=VERSION && export NCCL_SOCKET_IFNAME=eth0 && export NCCL_IB_HCA=eth0'
done
