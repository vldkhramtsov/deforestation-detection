#! /bin/bash

data_path=../data/diff_conc #_nohistmatch
image_size=56

neighbours=3

epochs=3
lr=1e-2
image_size=56

network=unet18
loss=bce_dice
optimizer=Adam

name="attention_"$network"_"$optimizer"_"$loss"_"$lr
echo "$name"

#_='
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
 			    --classification_head y 
 			    #--val_df $data_path/onlymasksplit/valid_df.csv \
                #--model_weights_path ../logs/$name_pretrain/checkpoints/best.pth
#'


#./run_tta_diff.sh $name $network $data_path $image_size $neighbours

#python evaluation_new.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/test_df.csv --output_name $name'_test'
#python evaluation_new.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/train_df.csv --output_name $name'_train'
#python evaluation_new.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/valid_df.csv --output_name $name'_val'

