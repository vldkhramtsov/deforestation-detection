#! /bin/bash

name=vld_unet50_1
echo "$name"
python train.py --lr 1e-2 --network unet50 --name $name
python prediction.py --data_path ../data/input/ --model_weights_path ../logs/$name/checkpoints/best.pth --test_df ../data/test_df.csv --save_path ../data/
python evaluation.py --datasets_path ../data/input --prediction_path ../data/predictions/ --test_df_path ../data/test_df.csv --output_name $name'_eval'

name=vld_unet50_2
echo "$name"
python train.py --lr 1e-3 --network unet50 --name $name
python prediction.py --data_path ../data/input/ --model_weights_path ../logs/$name/checkpoints/best.pth --test_df ../data/test_df.csv --save_path ../data/
python evaluation.py --datasets_path ../data/input --prediction_path ../data/predictions/ --test_df_path ../data/test_df.csv --output_name $name'_eval'

name=vld_unet50_3
echo "$name"
python train.py --lr 1e-4 --network unet50 --name $name
python prediction.py --data_path ../data/input/ --model_weights_path ../logs/$name/checkpoints/best.pth --test_df ../data/test_df.csv --save_path ../data/
python evaluation.py --datasets_path ../data/input --prediction_path ../data/predictions/ --test_df_path ../data/test_df.csv --output_name $name'_eval'

name=vld_unet50_4
echo "$name"
python train.py --lr 1e-3 --network unet50 --name $name --optimizer SGD
python prediction.py --data_path ../data/input/ --model_weights_path ../logs/$name/checkpoints/best.pth --test_df ../data/test_df.csv --save_path ../data/
python evaluation.py --datasets_path ../data/input --prediction_path ../data/predictions/ --test_df_path ../data/test_df.csv --output_name $name'_eval'


name=vld_unet101_1
echo "$name"
python train.py --lr 1e-2 --network unet101 --name $name
python prediction.py --data_path ../data/input/ --model_weights_path ../logs/$name/checkpoints/best.pth --test_df ../data/test_df.csv --save_path ../data/
python evaluation.py --datasets_path ../data/input --prediction_path ../data/predictions/ --test_df_path ../data/test_df.csv --output_name $name'_eval'

name=vld_unet101_2
echo "$name"
python train.py --lr 1e-3 --network unet101 --name $name
python prediction.py --data_path ../data/input/ --model_weights_path ../logs/$name/checkpoints/best.pth --test_df ../data/test_df.csv --save_path ../data/
python evaluation.py --datasets_path ../data/input --prediction_path ../data/predictions/ --test_df_path ../data/test_df.csv --output_name $name'_eval'

name=vld_unet101_3
echo "$name"
python train.py --lr 1e-4 --network unet101 --name $name
python prediction.py --data_path ../data/input/ --model_weights_path ../logs/$name/checkpoints/best.pth --test_df ../data/test_df.csv --save_path ../data/
python evaluation.py --datasets_path ../data/input --prediction_path ../data/predictions/ --test_df_path ../data/test_df.csv --output_name $name'_eval'

name=vld_unet101_4
echo "$name"
python train.py --lr 1e-3 --network unet101 --name $name --optimizer SGD
python prediction.py --data_path ../data/input/ --model_weights_path ../logs/$name/checkpoints/best.pth --test_df ../data/test_df.csv --save_path ../data/
python evaluation.py --datasets_path ../data/input --prediction_path ../data/predictions/ --test_df_path ../data/test_df.csv --output_name $name'_eval'


rm ../data/predictions/*.png
