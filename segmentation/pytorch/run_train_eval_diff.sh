#! /bin/bash

data_path=../data/diff5
image_size=56

epochs=200
lr=1e-3
image_size=56

network=unet18
loss=bce_dice
optimizer=RAdam

name="nobag"_$network"_"$optimizer"_"$loss"_"$lr
echo "$name"

python train.py --epochs $epochs \
                --image_size $image_size \
 			    --lr $lr \
 			    --network $network \
 			    --optimizer $optimizer \
 			    --loss $loss \
 			    --name $name \
 			    --dataset_path $data_path/ \
 			    --train_df $data_path/train_df.csv \
 			    --val_df $data_path/onlymasksplit/valid_df.csv \
                #--model_weights_path ../logs/$name_pretrain/checkpoints/best.pth


./run_tta_diff.sh $name $network $data_path $image_size
