for x in {1..40}
do
  echo "$x"
  python gen_hosts.py $x
  mpirun -x FI_PROVIDER="efa" -x FI_OFI_RXR_RX_COPY_UNEXP=1 -x FI_OFI_RXR_RX_COPY_OOO=1 -x FI_EFA_MR_CACHE_ENABLE=1 -x FI_OFI_RXR_INLINE_MR_ENABLE=1 -x NCCL_TREE_THRESHOLD=4294967296 --hostfile host$x -n 256 -N 8 --mca plm_rsh_no_tree_spawn 1 --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none -x NCCL_DEBUG=VERSION /home/ec2-user/nccl-tests/build/all_reduce_perf -b 642M -e 642M -f 2 -g 1 -c 1 -n 10 > op$x.out
  echo "Done $x"

done
