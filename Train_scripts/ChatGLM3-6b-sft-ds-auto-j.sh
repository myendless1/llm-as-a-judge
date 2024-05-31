NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" \
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file /vepfs-sha/liying/LLM_JUDGER/LLaMA-Factory/examples/accelerate/single_config_2.yaml --main_process_port 29502 \
    LLaMA-Factory/src/train.py Configs/chatglm3_6b_lora_sft_ds_auto_j.yaml