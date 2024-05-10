#!/bin/bash

VERSION="v1"
API_KEY="fd-cAgedppV6eQFfrD3mQgKVeSH6EMWDXFe7rvjYsyNzybG4RQ"
API_BASE="http://192.168.232.199:40035/v1"
MODEL="text-embedding-ada-002"
# https://huggingface.co/datasets/learnanything/sharegpt_v3_unfiltered_cleaned_split
DATASET="/home/jetson/benchmark/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
NUM_PROMPTS=1000
RATE="0.01"
#SEED=$((RANDOM % 101))
SAVE_RESULT=True
PARALLELS=(1 2 4 8 16 32 64)

for i in {1..100000}
do
  for PARALLEL in "${PARALLELS[@]}"
  do
  SEED=$((RANDOM % 101))
  /home/ubuntu/miniconda3/envs/test-env/bin/python \
  /home/ubuntu/benchmark/main_embedding.py.py \
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
