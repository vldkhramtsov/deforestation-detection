#! /bin/bash

name=$1
echo "$name"
mkdir ../data/predictions/$name
python prediction.py --data_path ../data/diff --model_weights_path ../logs/$name/checkpoints/best.pth --test_df ../data/diff/onlymasksplit/test_df.csv --save_path ../data/predictions/$name --network unet50 --tta True --merge_mode tsharping
python evaluation_new.py --datasets_path ../data/diff --prediction_path ../data/predictions/$name/predictions --test_df_path ../data/diff/onlymasksplit/test_df.csv --output_name $name'_eval'
python ../logs/plot_test_dice.py --prediction_path ../data/predictions/$name --test_df_path ../data/predictions/$name/predictionstest_df_results.csv
