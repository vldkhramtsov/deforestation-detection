#! /bin/bash

name=$1
network=$2
data_path=$3
image_size=$4
head=$5
echo "$name"
mkdir ../data/predictions/$name

echo "Test"
python prediction.py --classification_head $head --neighbours 3 --channels rgb b8 b8a b11 b12 ndvi ndmi --data_path $data_path --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/test_df.csv --save_path ../data/predictions/$name --network $network --size $image_size

echo "Train"
python prediction.py --classification_head $head --neighbours 3 --channels rgb b8 b8a b11 b12 ndvi ndmi --data_path $data_path --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/train_df.csv --save_path ../data/predictions/$name --network $network --size $image_size

echo "Valid"
python prediction.py --classification_head $head --neighbours 3 --channels rgb b8 b8a b11 b12 ndvi ndmi --data_path $data_path --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/valid_df.csv --save_path ../data/predictions/$name --network $network --size $image_size
