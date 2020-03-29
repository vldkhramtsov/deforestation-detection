#! /bin/bash

name=$1
network=$2
data_path=$3
image_size=$4
neighbours=$5

echo "$name"
mkdir ../data/predictions/$name
echo "Test onlymask"
python prediction.py --data_path $data_path --channels rgb b8 b8a b11 b12 ndvi ndmi --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/onlymasksplit/test_df.csv --save_path ../data/predictions/$name --network $network --size $image_size --neighbours $neighbours
python evaluation_new.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/onlymasksplit/test_df.csv --output_name 'test_onlymask'

echo "Train onlymask"
python prediction.py --data_path $data_path --channels rgb b8 b8a b11 b12 ndvi ndmi --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/onlymasksplit/train_df.csv --save_path ../data/predictions/$name --network $network --size  $image_size --neighbours $neighbours
python evaluation_new.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/onlymasksplit/train_df.csv --output_name 'train_onlymask'
echo "Valid onlymask"
python prediction.py --data_path $data_path --channels rgb b8 b8a b11 b12 ndvi ndmi --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/onlymasksplit/valid_df.csv --save_path ../data/predictions/$name --network $network --size $image_size --neighbours $neighbours
python evaluation_new.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/onlymasksplit/valid_df.csv --output_name 'val_onlymask'

echo "Test"
python prediction.py --data_path $data_path --channels rgb b8 b8a b11 b12 ndvi ndmi --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/test_df.csv --save_path ../data/predictions/$name --network $network --size $image_size --neighbours $neighbours
python evaluation_new.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/test_df.csv --output_name 'test'

echo "Train"
python prediction.py --data_path $data_path --channels rgb b8 b8a b11 b12 ndvi ndmi --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/train_df.csv --save_path ../data/predictions/$name --network $network --size $image_size --neighbours $neighbours
python evaluation_new.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/train_df.csv --output_name 'train'
echo "Valid"
python prediction.py --data_path $data_path --channels rgb b8 b8a b11 b12 ndvi ndmi --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/valid_df.csv --save_path ../data/predictions/$name --network $network --size $image_size --neighbours $neighbours
python evaluation_new.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/valid_df.csv --output_name 'val'

#python ./plot_prediction.sh $name $network $data_path $image_size  #--datasets_path $data_path/ --prediction_path ../data/predictions/$name --test_df_path "../data/predictions/$name/predictions"$name"_train_results.csv" #predictionstest_df_results.csv
#python ./plot_prediction.sh $name $network $data_path $image_size  #--datasets_path $data_path/ --prediction_path ../data/predictions/$name --test_df_path "../data/predictions/$name/predictions"$name"_test_results.csv"
#python ./plot_prediction.sh $name $network $data_path $image_size  #--datasets_path $data_path/ --prediction_path ../data/predictions/$name --test_df_path "../data/predictions/$name/predictions"$name"_val_results.csv"


python ../logs/plot_test_dice.py --datasets_path $data_path/ --prediction_path ../data/predictions/$name --test_df_path "../data/predictions/$name/predictions"$name"_train_results.csv" #predictionstest_df_results.csv
python ../logs/plot_test_dice.py --datasets_path $data_path/ --prediction_path ../data/predictions/$name --test_df_path "../data/predictions/$name/predictions"$name"_test_results.csv"
python ../logs/plot_test_dice.py --datasets_path $data_path/ --prediction_path ../data/predictions/$name --test_df_path "../data/predictions/$name/predictions"$name"_val_results.csv"

