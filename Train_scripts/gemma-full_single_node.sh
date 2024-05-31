#!/bin/bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
NPROC_PER_NODE=2
NNODES=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/train.py Configs/gemme-2b_full.yaml
