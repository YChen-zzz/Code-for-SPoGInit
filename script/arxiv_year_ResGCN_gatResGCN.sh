#!/bin/bash

for model in ResGCN gatResGCN
do
for depth in 32
do

cd SPoGInit

data_path=..
save_path=..


python ./Arxiv_run.py \
    --seed 42 \
    --num_layers $depth \
    --model $model \
    --dropout 0.5 \
    --activation ReLU \
    --data arxiv-year \
    --runs 5 \
    --data_path $data_path \
    --save_path $save_path \
    --initialization conventional \
    --hidden 64 \
    --epoch 1000 

python ./Arxiv_run.py \
    --seed 42 \
    --num_layers $depth \
    --model $model \
    --dropout 0.5 \
    --activation ReLU \
    --data arxiv-year \
    --runs 5 \
    --data_path $data_path \
    --save_path $save_path \
    --initialization glorot \
    --hidden 64 \
    --epoch 1000 

# add use_spog to use SPoGInit
python ./Arxiv_run.py \
    --seed 42 \
    --num_layers $depth \
    --model $model \
    --dropout 0.5 \
    --activation ReLU \
    --data arxiv-year \
    --runs 5 \
    --data_path $data_path \
    --save_path $save_path \
    --initialization glorot \
    --use_spog \
    --hidden 64 \
    --epoch 1000 
done
done
