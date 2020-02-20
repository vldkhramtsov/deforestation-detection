#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 23:30:23 2020

@author: vld-kh
"""

import os
import csv
import imageio
import datetime
import argparse

import numpy as np
import pandas as pd
import rasterio as rs
import matplotlib.pyplot as plt

from tqdm import tqdm
from random import random
from rasterio.plot import reshape_as_image as rsimg

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating difference between images in the near-by time series of one tile.'
    )
    parser.add_argument(
        '--data_path', '-dp', dest='data_path',
        default='../data/input', help='Path to input data'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../data/data_diff/input', 
        help='Path to directory where pieces will be stored'
    )
    parser.add_argument(
        '--img_path', '-ip', dest='img_path',
        default='images', help='Path to pieces of image'
    )
    parser.add_argument(
        '--msk_path', '-mp', dest='msk_path',
        default='masks', help='Path to pieces of mask'
    )
    parser.add_argument(
        '--train_size', '-tr', dest='train_size',
        default=0.6, type=float, help='Represent proportion of the dataset to include in the train split'
    )
    parser.add_argument(
        '--test_size', '-ts', dest='test_size',
        default=0.2, type=float, help='Represent proportion of the dataset to include in the test split'
    )
    parser.add_argument(
        '--valid_size', '-vl', dest='valid_size',
        default=0.2, type=float, help='Represent proportion of the dataset to include in the valid split'
    )
    return parser.parse_args()

def getdates(data_path):
    tiles = [ [name, datetime.datetime.strptime(name[-15:-11]+'-'+name[-11:-9]+'-'+name[-9:-7], 
        '%Y-%m-%d')] for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name)) ]
    return tiles

def readtiff(filename):
    src = rs.open(filename)
    return rsimg(src.read()), src.meta

def imgdiff(tile1, tile2, diff_path, data_path, img_path, msk_path, writer):
    xs = [piece.split('_')[4:5][0] for piece in os.listdir(os.path.join(data_path,tile1,img_path))]
    ys = [piece.split('_')[5:6][0].split('.')[0] for piece in os.listdir(os.path.join(data_path,tile1,img_path))]
    assert len(xs)==len(ys)
    for i in range(len(xs)):
        img1,meta = readtiff( 
                        os.path.join(data_path,tile1,img_path,tile1+'_'+xs[i]+'_'+ys[i]+'.tiff') )
        img2, _   = readtiff( 
                        os.path.join(data_path,tile2,img_path,tile2+'_'+xs[i]+'_'+ys[i]+'.tiff') ) 
        
        msk1=imageio.imread(
                        os.path.join(data_path,tile1,msk_path,tile1+'_'+xs[i]+'_'+ys[i]+'.png'))
        msk2=imageio.imread(
                        os.path.join(data_path,tile2,msk_path,tile2+'_'+xs[i]+'_'+ys[i]+'.png'))
        
        diff_img = np.clip((img1-img2), 0, 255)
        diff_msk = np.clip((msk1-msk2), 0, 255)
        
        with rs.open(os.path.join(diff_path, img_path, diff_path.split('/')[-1]+'_'+xs[i]+'_'+ys[i]+'.tiff'), 'w', **meta) as dst:
            for ix in range(diff_img.shape[2]):
                dst.write(diff_img[:, :, ix], ix + 1)
        dst.close()

        imageio.imwrite(os.path.join(diff_path, msk_path, diff_path.split('/')[-1]+'_'+xs[i]+'_'+ys[i]+'.png'), diff_msk)
        writer.writerow([
            diff_path.split('/')[-1], diff_path.split('/')[-1], xs[i]+'_'+ys[i], int(diff_msk.sum()/255)
        ])

    
def get_diff_and_split(data_path, save_path, img_path, msk_path, train_size, test_size, valid_size):
    tiles=getdates(data_path)
    df = pd.DataFrame(tiles, columns=['tileID','img_date'])
    df = df.sort_values(['img_date'],ascending=False)
    
    infofile=os.path.join(save_path,'data_info.csv')    
    with open(infofile, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([
            'dataset_folder', 'name', 'position', 'mask_pxl'
        ])
        for i in tqdm(range(len(df)-1)):
            j=i+1
            diff_path = os.path.join(save_path, str(df['img_date'].iloc[i].date())+'_'+str(df['img_date'].iloc[j].date()))
            if not os.path.exists(diff_path):
                os.mkdir(diff_path)
            if not os.path.exists(os.path.join(diff_path,img_path)):
                os.mkdir(os.path.join(diff_path,img_path))
            if not os.path.exists(os.path.join(diff_path,msk_path)):
                os.mkdir(os.path.join(diff_path,msk_path))
            
            imgdiff(df['tileID'].iloc[i], df['tileID'].iloc[j],diff_path,data_path, img_path, msk_path,writer)
            
    df = pd.read_csv(infofile)
    xy = df['position'].unique()
    
    np.random.seed(seed=59)
    rand = np.random.random(size=len(xy))
    
    train=[]
    test=[]
    valid=[]
    for i in range(len(xy)):
        if rand[i]<=train_size:
            train.append(xy[i])
        elif rand[i]>train_size and rand[i]<train_size+test_size:
            test.append(xy[i])
        else:
            valid.append(xy[i])
    
    for data_type, name_type in zip([train,test,valid],['train','test','valid']):
        output_file=os.path.join(save_path,f'{name_type}_df.csv')
        os.system(f'head -n1 {infofile} > {output_file}')
        for position in data_type:
            df[df['position']==position].to_csv(output_file,mode='a',header=False,index=False,sep=',')
    print('Train split: %d'%len(train))
    print('Test  split: %d'%len(test))
    print('Valid split: %d'%len(valid))

if __name__ == '__main__':
    args = parse_args()
    assert args.train_size + args.test_size + args.valid_size==1.0
    get_diff_and_split(args.data_path, args.save_path, args.img_path, args.msk_path, 
                       args.train_size, args.test_size, args.valid_size)
