#!/bin/bash

layer_list=(4 8 16 32 64)
lr_list=(0.005 0.005 0.0005 0.0005 0.00005)
dropout_list=(0.5 0.5 0 0 0)

cd SPoGInit

data_path=..
save_path=..

for model in GCN
do
for i in 2
do
python ./Arxiv_run.py \
    --seed 42 \
    --num_layers ${layer_list[$i]} \
    --model $model \
    --dropout ${dropout_list[$i]} \
    --lr ${lr_list[$i]} \
    --activation Tanh \
    --runs 3 \
    --data_path $data_path \
    --save_path $save_path \
    --initialization conventional \
    --hidden 64 \
    --epoch 1000 

python ./Arxiv_run.py \
    --seed 42 \
    --num_layers ${layer_list[$i]} \
    --model $model \
    --dropout ${dropout_list[$i]} \
    --lr ${lr_list[$i]} \
    --activation Tanh \
    --runs 3 \
    --data_path $data_path \
    --save_path $save_path \
    --initialization glorot \
    --hidden 64 \
    --epoch 1000 

# add use_spog to use SPoGInit
python ./Arxiv_run.py \
    --seed 42 \
    --num_layers ${layer_list[$i]} \
    --model $model \
    --dropout ${dropout_list[$i]} \
    --lr ${lr_list[$i]} \
    --activation Tanh \
    --runs 3 \
    --data_path $data_path \
    --save_path $save_path \
    --initialization glorot \
    --use_spog \
    --hidden 64 \
    --epoch 1000 
done
done
