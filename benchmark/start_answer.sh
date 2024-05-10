#!/bin/bash

VERSION="v1"
API_KEY="fd-cAgedppV6eQFfrD3mQgKVeSH6EMWDXFe7rvjYsyNzybG4RQ"
API_BASE="http://192.168.184.69:40077/v1"
MODEL="jfh-qwen-14b-chat"
# https://huggingface.co/datasets/ceval/ceval-exam
# val dataset
DATASET="/home/ubuntu/benchmark/val"
NUM_PROMPTS=1000
RATE="0.1"
SEED=$((RANDOM % 101))
SAVE_RESULT=True
PARALLELS=(1 2 4 8 16 32 64)

for i in {1..10}
do
for PARALLEL in "${PARALLELS[@]}"
do
/home/ubuntu/miniconda3/envs/test-env/bin/python \
/home/ubuntu/benchmark/main_chat_answer.py \
--version=$VERSION \
--api-key=$API_KEY \
--api-base=$API_BASE \
--model=$MODEL \
--dataset=$DATASET \
--num-prompts=$NUM_PROMPTS \
--seed=$SEED \
--parallel=$PARALLEL \
--save-result
done
done