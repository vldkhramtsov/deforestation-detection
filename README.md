# Deforestation Detection  

## Project structure info
 * `input` - scripts for data download and preparation
 * `segmentation` - investigation about model approach, model training and model evaluation of clearcut detection

## Credential setup

This project needs several secure credentials, for peps.cnes.fr and sentinel-hub. 
For correct setup, you need to create peps_download_config.ini 
(it could be done by example peps_download_config.ini.example) and feel auth, 
password and sentinel_id parameters.

## Model Development Guide
### Data downloading
1) Create an account on https://peps.cnes.fr/rocket/#/home

2) Specify params in config file input/peps_download_config.ini

3) Download an image archive `python peps_download.py`

4) Unzip the archive

5) Merge bands with `python prepare_tif.py --data_folder … --save_path …` for a single image folder, or `./PREPARE_IMAGES.sh "data_folder" "save_path"` for the catalogue of images.

6) Download global land cover map: `wget https://s3-eu-west-1.amazonaws.com/vito.landcover.global/2015/E020N60_ProbaV_LC100_epoch2015_global_v2.0.2_products_EPSG-4326.zip`
    * Unzip archive
    * Run script `python prepare_landcover.py --save_path ... --data_path .../E020N60_ProbaV_LC100_epoch2015_global_v2.0.2_discrete-classification_EPSG-4326.tif`

### Data preparation
1) Create folder in clearcut_research where is stored data:
   * Source subfolder stores raw data that has to be preprocess
   * Input subfolder stores data that is used in training and evaluation
   * Polygons subfolder stores markup

2) The source folder contains folders for each image that you downloaded. In that folder you have to store TIFF images of channels (in our case RGB, B2 and B8) named as f”{image_folder}\_{channel}.tif”.

3) If you have already merged bands to a single TIFF, you can just move it to input folder. But you have to create the folder (it can be empty) for this image in the source folder.

4) The polygons folder contains markup that you apply to all images in input folder.

#### Example of data folder structure:
```
data
├── input
│   ├── image0.tif
│   └── image1.tif
├── polygons
│   └── markup.geojson
└── source
    ├── image0
    │   ├── image0_b2.tif
    │   ├── image0_b8.tif
    │   └── image0_rgb.tif
    └── image1
        ├── iamge1_b2.tif
        ├── image1_b8.tif
        └── image1_rgb.tif
```
5) Run preprocessing on this data. You can specify other params if it necessary (add --no_merge if you have already merged channels with prepare_tif.py script).
```
python preprocessing.py \
 --polys_path ../data/polygons/markup.geojson \
 --tiff_path ../data/source
 --save_path ../data/input
 --land_path ../data/auxiliary
```

#### Example of input folder structure after preprocessing:
```
input
├── image0
│   ├── geojson_polygons
│   ├── image0.png
│   ├── image_pieces.csv
│   ├── images
│   ├── instance_masks
│   ├── masks
│   └── landcover
├── image0.tif
├── image1
│   ├── geojson_polygons
│   ├── image1.png
│   ├── image_pieces.csv
│   ├── images
│   ├── instance_masks
│   └── masks
└── image1.tif
```
6) Run data division script with specified split_function (default=’geo_split’) to create train/test/val datasets.
```
python generate_data.py --markup_path ../data/polygons/markup.geojson
```

### Model training
1) If it necessary specify augmentation in pytorch/dataset.py

2) Specify hyperparams in pytorch/train.py

3) Run training `python train.py`

### Model evaluation
1) Generate predictions 
```
python prediction.py \
 --data_path ../data/input \
 --model_weights_path … \
 --test_df ../data/test_df.csv \
 --save_path ../data
```  
2) Run evaluation
```
python evaluation.py \
 --datasets_path ../data/input \
 --prediction_path ../data/predictions \
 --test_df_path ../data/test_df.csv \
 --output_name …
```
3) Run notebooks/tf_records_visualizer.ipynb to view results of evaluation.

4) Run notebooks/f1_score.ipynb to get metrics for the whole image.
