#! /bin/bash
name=$1

python ../logs/plot_test_dice.py --datasets_path $data_path/ --prediction_path ../data/predictions/$name --test_df_path "../data/predictions/$name/predictions"$name"_train_results.csv" #predictionstest_df_results.csv
python ../logs/plot_test_dice.py --datasets_path $data_path/ --prediction_path ../data/predictions/$name --test_df_path "../data/predictions/$name/predictions"$name"_test_results.csv"
python ../logs/plot_test_dice.py --datasets_path $data_path/ --prediction_path ../data/predictions/$name --test_df_path "../data/predictions/$name/predictions"$name"_val_results.csv"
