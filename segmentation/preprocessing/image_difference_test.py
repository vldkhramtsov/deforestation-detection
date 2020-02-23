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
from skimage import img_as_ubyte
from rasterio.plot import reshape_as_image as rsimg

def getdates(data_path):
    tiles = [ [name, datetime.datetime.strptime(name[-15:-11]+'-'+name[-11:-9]+'-'+name[-9:-7], 
        '%Y-%m-%d')] for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name)) ]
    return tiles

def readtiff(filename):
    src = rs.open(filename)
    return rsimg(src.read()), src.meta

def diff(img1,img2):
    #                         [-1,1]               ---------------> [0,2]->[0,1]
    #d = ((img1.astype(np.float32) - img2.astype(np.float32)) / 255 + 1 ) / 2
    d = img1.astype(np.float32) - img2.astype(np.float32)
    return img_as_ubyte(  (d-np.min(d))/(np.max(d)-np.min(d)) )

def imgdiff(tile1, tile2, diff_path, data_path, img_path, msk_path):
    xs = [piece.split('_')[4:5][0] for piece in os.listdir(os.path.join(data_path,tile1,img_path))]
    ys = [piece.split('_')[5:6][0].split('.')[0] for piece in os.listdir(os.path.join(data_path,tile1,img_path))]
    assert len(xs)==len(ys)
    for i in range(len(xs)):
        img1,meta = readtiff( 
                        os.path.join(data_path,tile1,img_path,tile1+'_'+xs[i]+'_'+ys[i]+'.tiff') )
        img2, _   = readtiff( 
                        os.path.join(data_path,tile2,img_path,tile2+'_'+xs[i]+'_'+ys[i]+'.tiff') ) 
        meta['dtype'] = np.float32
        
        msk1=imageio.imread(
                        os.path.join(data_path,tile1,msk_path,tile1+'_'+xs[i]+'_'+ys[i]+'.png'))
        msk2=imageio.imread(
                        os.path.join(data_path,tile2,msk_path,tile2+'_'+xs[i]+'_'+ys[i]+'.png'))
        
        diff_img = np.clip((img1-img2), 0, 255)
        diff_msk = np.clip((msk1-msk2), 0, 255)
        if(diff_msk.sum()>0):
            plt.figure(figsize=(12,18))
            lsize=10
            j=1
            for k in range(5):
                ax = plt.subplot(6,3,j)
                im = ax.imshow(img1[:,:,k])
                cb=plt.colorbar(im,ax=ax)
                cb.ax.tick_params(labelsize=lsize)
                j+=1
                
                ax = plt.subplot(6,3,j)
                im = ax.imshow(img2[:,:,k])
                cb=plt.colorbar(im,ax=ax)
                cb.ax.tick_params(labelsize=lsize)
                j+=1
                
                ax = plt.subplot(6,3,j)
                im = ax.imshow(diff(img1[:,:,k],img2[:,:,k]))
                cb=plt.colorbar(im,ax=ax)
                cb.ax.tick_params(labelsize=lsize)
                j+=1
    
            ax = plt.subplot(6,3,j)
            im = ax.imshow(msk1)
            cb=plt.colorbar(im,ax=ax)
            cb.ax.tick_params(labelsize=lsize)
    
            ax = plt.subplot(6,3,j+1)
            im = ax.imshow(msk2)
            cb=plt.colorbar(im,ax=ax)
            cb.ax.tick_params(labelsize=lsize)
    
            ax = plt.subplot(6,3,j+2)
            im = ax.imshow(msk1-msk2)
            cb=plt.colorbar(im,ax=ax)
            cb.ax.tick_params(labelsize=lsize)
    
            plt.show()
            #plt.savefig(f'./tmp/{xs[i]}_{ys[i]}.png')
            plt.close()
    
def get_diff_and_split(data_path, save_path, img_path, msk_path):
    tiles=getdates(data_path)
    df = pd.DataFrame(tiles, columns=['tileID','img_date'])
    df = df.sort_values(['img_date'],ascending=False)
    
    i=0
    j=i+1
    diff_path = os.path.join(save_path, str(df['img_date'].iloc[i].date())+'_'+str(df['img_date'].iloc[j].date()))    
    imgdiff(df['tileID'].iloc[i], df['tileID'].iloc[j],diff_path, data_path, img_path, msk_path)
#            
#    df = pd.read_csv(infofile)
#    xy = df['position'].unique()
#    
#    np.random.seed(seed=59)
#    rand = np.random.random(size=len(xy))
#    
#    train=[]
#    test=[]
#    valid=[]
#    for i in range(len(xy)):
#        if rand[i]<=train_size:
#            train.append(xy[i])
#        elif rand[i]>train_size and rand[i]<train_size+test_size:
#            test.append(xy[i])
#        else:
#            valid.append(xy[i])
#    
#    for data_type, name_type in zip([train,test,valid],['train','test','valid']):
#        output_file=os.path.join(save_path,f'{name_type}_df.csv')
#        os.system(f'head -n1 {infofile} > {output_file}')
#        for position in data_type:
#            df[df['position']==position].to_csv(output_file,mode='a',header=False,index=False,sep=',')
#    print('Train split: %d'%len(train))
#    print('Test  split: %d'%len(test))
#    print('Valid split: %d'%len(valid))

data_path='../data/input'
save_path=''
img_path='images'
msk_path='masks'

get_diff_and_split(data_path, save_path, img_path, msk_path)