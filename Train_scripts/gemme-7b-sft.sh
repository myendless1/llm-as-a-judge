export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train Configs/gemme-7b_lora_sft.yaml

NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file /vepfs-sha/liying/LLM_JUDGER/LLaMA-Factory/examples/accelerate/single_config_2.yaml \
    LLaMA-Factory/src/train.py  Configs/gemme-7b_lora_sft.yaml

NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file /vepfs-sha/liying/LLM_JUDGER/LLaMA-Factory/examples/accelerate/single_config_2.yaml \
    LLaMA-Factory/src/train.py Configs/gemme-2b_full.yaml