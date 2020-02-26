#! /bin/bash

optimizer=RAdam
network=fpn50

name_pretrain=pretrain
echo "$name_pretrain"
python train.py --epochs 120 \
 			    --lr 9e-2 \
 			    --network $network \
 			    --optimizer $optimizer \
 			    --name $name_pretrain \
 			    --dataset_path ../data/diff/ \
 			    --train_df ../data/diff/onlymasksplit/train_df.csv \
 			    --val_df ../data/diff/onlymasksplit/valid_df.csv \

epochs=60
lr=1e-3
loss=bce
optimizer=RAdam
network=fpn50

name="diff_"$network"_"$optimizer"_"$loss"_"$lr
echo "$name"

python train.py --epochs $epochs \
 			    --lr $lr \
 			    --network $network \
 			    --optimizer $optimizer \
 			    --loss $loss \
 			    --name $name \
 			    --dataset_path ../data/diff/ \
 			    --train_df ../data/diff/train_df_aug.csv \
 			    --val_df ../data/diff/valid_df.csv \
 			    --model_weights_path ../logs/$name_pretrain/checkpoints/best.pth


./run_tta_diff.sh $name
