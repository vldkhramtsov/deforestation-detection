import os
import cv2
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
from imgaug import augmenters as iaa
from rasterio.plot import reshape_as_image as rsimg

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for dividing images into smaller pieces.'
    )
    parser.add_argument(
        '--data_path', '-dp', dest='data_path',
        default='../data/input', help='Path to input data'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../data/diff2', 
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
        '--cld_path', '-mp', dest='cld_path',
        default='masks', help='Path to pieces of cloud map'
    )
    parser.add_argument(
        '--width', '-w',  dest='width', default=224,
        type=int, help='Width of a piece'
    )
    parser.add_argument(
        '--height', '-hgt', dest='height', default=224,
        type=int, help='Height of a piece'
    )
    parser.add_argument(
        '--neighbours', '-nbr', dest='neighbours', default=2,
        type=int, help='Number of pairs before the present day (dt=5 days, e.g. neighbours=3 means max dt = 15 days)'
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

def diff(img1,img2,width,height):
    dim = (width,height)
    I1 = np.clip(cv2.resize(img1.astype(np.float32) , dim, interpolation = cv2.INTER_CUBIC), 0, 255)
    I2 = np.clip(cv2.resize(img2.astype(np.float32) , dim, interpolation = cv2.INTER_CUBIC), 0, 255)
    d = ( (I1 - I2) / (I1 + I2) )
    #return img_as_ubyte(d)
    return ((d+1)*127).astype(np.uint8)


def imgdiff(tile1, tile2, diff_path, data_path, img_path, msk_path, cloud_path, writer, width,height):
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
        
        cld1=imageio.imread(
                        os.path.join(data_path,tile1,cloud_path,tile1+'_'+xs[i]+'_'+ys[i]+'.png'))
        cld2=imageio.imread(
                        os.path.join(data_path,tile2,cloud_path,tile2+'_'+xs[i]+'_'+ys[i]+'.png'))
        
        if (cld1/255+cld2/255).sum()<0.3*cld1.size():
	        diff_img = diff(img1,img2, width,height)
	        diff_msk = np.clip((msk1-msk2), 0, 255)
	        diff_msk = cv2.resize(diff_msk, (height,width), interpolation = cv2.INTER_NEAREST)
	        
	        meta['width'] = width
	        meta['height'] = height

	        with rs.open(os.path.join(diff_path, img_path, diff_path.split('/')[-1]+'_'+xs[i]+'_'+ys[i]+'.tiff'), 'w', **meta) as dst:
	            for ix in range(diff_img.shape[2]):
	                dst.write(diff_img[:, :, ix], ix + 1)
	        dst.close()

	        imageio.imwrite(os.path.join(diff_path, msk_path, diff_path.split('/')[-1]+'_'+xs[i]+'_'+ys[i]+'.png'), diff_msk)
	        writer.writerow([
	            diff_path.split('/')[-1], diff_path.split('/')[-1], xs[i]+'_'+ys[i], int(diff_msk.sum()/255)
	        ])
        else:
	        pass

def get_diff_and_split(data_path, save_path, img_path, msk_path, cloud_path, width, height, neighbours, train_size, test_size, valid_size):
    tiles=getdates(data_path)
    df = pd.DataFrame(tiles, columns=['tileID','img_date'])
    df = df.sort_values(['img_date'],ascending=False)
    
    infofile=os.path.join(save_path,'data_info.csv')    
    with open(infofile, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([
            'dataset_folder', 'name', 'position', 'mask_pxl'
        ])
        for i in range(len(df)-1):
            for j in range(i+1, i+1+neighbours):
                if(j<len(df)):
                    print(str(df['img_date'].iloc[i].date())+' - '+str(df['img_date'].iloc[j].date()))
                    diff_path = os.path.join(save_path, str(df['img_date'].iloc[i].date())+'_'+str(df['img_date'].iloc[j].date()))
                    if not os.path.exists(diff_path):
                        os.mkdir(diff_path)
                    if not os.path.exists(os.path.join(diff_path,img_path)):
                        os.mkdir(os.path.join(diff_path,img_path))
                    if not os.path.exists(os.path.join(diff_path,msk_path)):
                        os.mkdir(os.path.join(diff_path,msk_path))
                    
                    imgdiff(df['tileID'].iloc[i], df['tileID'].iloc[j],diff_path,
                    	data_path, img_path, msk_path,cloud_path,
                    	writer,width,height)
            
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

def augment_masked_images(data_path, save_path, img_path, msk_path, train_df_path):
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-25, 25)),
        iaa.Crop(percent=(0, 0.1)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.ElasticTransformation(alpha=3, sigma=1)
    ], random_order=True)
    
    train_df = pd.read_csv(train_df_path)
    aug_train_df_path = train_df_path.split(".csv")[-2]+'_aug.csv'
    os.system(f'cat {train_df_path} > {aug_train_df_path}')
    total_imgs = train_df.shape[0]
    train_df = train_df[train_df['mask_pxl']>0]
    masked_imgs = train_df.shape[0]
    number_of_augmentation = int(total_imgs/masked_imgs)+1
    print('number_of_augmentation:',number_of_augmentation)
    aug_path = os.path.join(save_path, 'augmented')
    if not os.path.exists(aug_path):
        os.mkdir(aug_path)
    if not os.path.exists(os.path.join(aug_path,img_path)):
        os.mkdir(os.path.join(aug_path,img_path))
    if not os.path.exists(os.path.join(aug_path,msk_path)):
        os.mkdir(os.path.join(aug_path,msk_path))
        
    for _, row in tqdm(train_df.iterrows()):
        image, meta = readtiff(os.path.join(save_path, row['dataset_folder'],img_path,row['name']+'_'+row['position']+'.tiff'))
        segmap = imageio.imread(os.path.join(save_path, row['dataset_folder'],msk_path,row['name']+'_'+row['position']+'.png'))
        
        images = np.zeros((number_of_augmentation, image.shape[0], image.shape[1], image.shape[2]))
        segmaps = np.zeros((number_of_augmentation, image.shape[0], image.shape[1], 1))
        for n in range(number_of_augmentation):
            images[n] = image
            segmaps[n] = segmap.reshape((image.shape[0], image.shape[1], 1))
        
        images = images.astype(np.uint8)
        segmaps = segmaps.astype(np.uint8)
        images_aug, segmaps_aug = seq(images=images, segmentation_maps=segmaps)
        
        for n in range(number_of_augmentation):
            img = images_aug[n].reshape((image.shape[0],image.shape[1],image.shape[2]))
            msk  =segmaps_aug[n].reshape((image.shape[0],image.shape[1]))
            with rs.open(os.path.join(aug_path,img_path,row['name']+'_'+str(n)+'_'+row['position']+'.tiff'), 'w', **meta) as dst:
                for ix in range(img.shape[2]):
                    dst.write(img[:, :, ix], ix + 1)
            dst.close() 
            imageio.imwrite(os.path.join(aug_path,msk_path,row['name']+'_'+str(n)+'_'+row['position']+'.png'), msk)
            row_info = pd.DataFrame(['augmented',  row['name']+'_'+str(n), row['position'], int(msk.sum()/255)]).T
            row_info.to_csv(f'{aug_train_df_path}', header=None, index=None, mode='a')
            

if __name__ == '__main__':
    args = parse_args()
    assert args.train_size + args.test_size + args.valid_size==1.0
    assert args.neighbours > 0 and args.neighbours < 4
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    get_diff_and_split(args.data_path, args.save_path, args.img_path, args.msk_path, args.cld_path,
                       args.width,args.height, args.neighbours,
                       args.train_size, args.test_size, args.valid_size)
    augment_masked_images(args.data_path, args.save_path, args.img_path, args.msk_path, 
                          os.path.join(args.save_path,'train_df.csv'))