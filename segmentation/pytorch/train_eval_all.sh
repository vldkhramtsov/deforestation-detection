#! /bin/bash

data_path=../data/siam

epochs=250
lr=1e-2
image_size=56
optimizer=Adam

loss=bce_dice
model=unet

name="siam"_$model"_"$optimizer"_"$loss"_"$lr
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


loss=tversky
model=unet3d

name="siam"_$model"_"$optimizer"_"$loss"_"$lr
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


loss=tversky
model=siamdiff

name="siam"_$model"_"$optimizer"_"$loss"_"$lr
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


loss=tversky
model=siamconc

name="siam"_$model"_"$optimizer"_"$loss"_"$lr
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




data_path=../data/diff_conc
image_size=56

neighbours=3

epochs=350
lr=1e-2
image_size=56

network=unet18
loss=bce_dice
optimizer=Adam

head=True

name="diff_"$network"_"$optimizer"_"$loss"_"$lr"_"$head
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
 			    --val_df $data_path/valid_df.csv \
 			    --channels rgb b8 b8a b11 b12 ndvi ndmi \
 			    --neighbours $neighbours \
 			    --classification_head $head

./run_pred_diff.sh $name $network $data_path $image_size $head
./run_eval_diff.sh $name $network $data_path $image_size
./run_plot_diff.sh $name


head=False

name="diff_"$network"_"$optimizer"_"$loss"_"$lr"_"$head
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
 			    --val_df $data_path/valid_df.csv \
 			    --channels rgb b8 b8a b11 b12 ndvi ndmi \
 			    --neighbours $neighbours \
 			    --classification_head $head

./run_pred_diff.sh $name $network $data_path $image_size $head
./run_eval_diff.sh $name $network $data_path $image_size
./run_plot_diff.sh $name


