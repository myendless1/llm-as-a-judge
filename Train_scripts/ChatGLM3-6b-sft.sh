export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train Configs/chatglm3_6b_lora_sft.yaml  --use_fast_tokenizer False
# CUDA_VISIBLE_DEVICES=1 llamafactory-cli webchat Configs/inference_chatglm_lora_sft.yaml