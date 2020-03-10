import os
import imageio
import argparse

import numpy as np
import pandas as pd
import rasterio as rs
import matplotlib.pyplot as plt

from tqdm import tqdm
from rasterio.plot import reshape_as_image as rsimg

import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 40})
def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for evaluating performance of the model.')
    parser.add_argument(
        '--datasets_path', '-dp', dest='datasets_path',
        default='../data/diff', help='Path to the directory all the data')
    parser.add_argument(
        '--prediction_path', '-pp', dest='prediction_path',
        default='../data/predictions/vld_unet50_1_diff', help='Path to the directory with predictions')
    parser.add_argument(
        '--test_df_path', '-tp', dest='test_df_path',
        default='../data/predictions/vld_unet50_1_diff/predictionstest_df_results.csv', help='Path to the test dataframe with image names')
    parser.add_argument(
        '--images_folder', '-imf', dest='images_folder',
        default='images',
        help='Name of folder where images are storing'
    )
    parser.add_argument(
        '--masks_folder', '-mf', dest='masks_folder',
        default='masks',
        help='Name of folder where masks are storing'
    )
    return parser.parse_args()

def readtiff(filename):
    src = rs.open(filename)
    return rsimg(src.read())

def PlotPredictions(test_df_path, datasets_path, prediction_path, images_folder, masks_folder): 
    test_df = pd.read_csv(test_df_path)
    save_path = os.path.join(prediction_path, 'result', test_df_path.split('/')[-1])
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print("Result directory created.")
    for _, image_info in tqdm(test_df.iterrows()):
        img_file = os.path.join(datasets_path,
                                image_info['dataset_folder'],
                                images_folder,
                                image_info['name']+'.tiff')
        img = readtiff(img_file)
        
        mask_file = os.path.join(datasets_path,
                                image_info['dataset_folder'],
                                masks_folder,
                                image_info['name']+'.png')        
        
        mask = imageio.imread(mask_file)
        
        pred_file = os.path.join(prediction_path,
                                'predictions',
                                image_info['name']+'.png')        
        
        pred = imageio.imread(pred_file)
        
        plt.figure(figsize=(35,6))
        
        for i in range(img.shape[2]):
            ax=plt.subplot(1,img.shape[2]+2,i+1)
            ax.imshow(img[:,:,i])
            ax.axis('off')
        
        ax.set_title('Dice score: %.4f'%(image_info['dice_score']))
        ax=plt.subplot(1,img.shape[2]+2,i+2)
        ax.imshow(mask, cmap='RdBu_r')
        ax.axis('off')
        ax=plt.subplot(1,img.shape[2]+2,i+3)
        ax.imshow(pred, cmap='RdBu_r')
        ax.axis('off')
        plt.savefig(os.path.join(save_path, image_info['name']+'.png'))
        plt.close()
        
        
        
        
        
if __name__ == '__main__':
    args = parse_args()
    PlotPredictions(args.test_df_path, args.datasets_path, args.prediction_path,
                    args.images_folder, args.masks_folder)
    
        
