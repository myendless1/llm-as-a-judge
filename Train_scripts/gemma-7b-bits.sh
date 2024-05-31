export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train Configs/gemma-7b-bitstand.yaml