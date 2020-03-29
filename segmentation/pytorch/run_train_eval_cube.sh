#! /bin/bash

data_path=../data/siam

epochs=1
lr=1e-2
image_size=56

loss=bce_dice
optimizer=Adam

model=unet3d

name="siam_corr"_$model"_"$optimizer"_"$loss"_"$lr
echo "$name"

python trainsiam.py --epochs $epochs \
                --image_size $image_size \
 			    --lr $lr \
 			    --model $model \
 			    --network unet18 \
 			    --optimizer $optimizer \
 			    --loss $loss \
 			    --name $name \
 			    --dataset_path $data_path/ \
 			    --train_df $data_path/train_df.csv \
 			    --val_df $data_path/onlymasksplit/valid_df.csv \
 			    --test_df $data_path/test_df.csv \
 			    --mode train

python trainsiam.py --epochs $epochs \
                --image_size $image_size \
 			    --lr $lr \
 			    --model $model \
 			    --network unet18 \
 			    --optimizer $optimizer \
 			    --loss $loss \
 			    --name $name \
 			    --dataset_path $data_path/ \
 			    --train_df $data_path/train_df.csv \
 			    --val_df $data_path/valid_df.csv \
 			    --test_df $data_path/test_df.csv \
 			    --mode eval
