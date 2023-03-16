import os, argparse, glob, time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from numpy.random import default_rng
from cellpose import models, utils, io
from imagegrains import segmentation_helper, grainsizing, gsd_uncertainty
from imagegrains import data_loader, plotting

def main():
    parser = argparse.ArgumentParser(description='ImageGrains')

    seg_args=parser.add_argument_group('Segmentation')
    seg_args.add_argument('--mute_output', default=None, type=bool, help='Mute console output. If True, example plots will not be saved.')
    seg_args.add_argument('--img_dir', default=None, type=str, help='Input directory for images to segment')
    seg_args.add_argument('--out_dir', default=None, type=str, help='Output directory for segmented images; if not specified, the images will be saved in the same directory as the input images')
    seg_args.add_argument('--img_type', default='jpg', type=str, help='Image type to segment; by default the the script will look for .jpg files. Alternatively, .tif and .png files can be segmented.')
    seg_args.add_argument('--model_dir', default=None, type=str, help='Segemntation model to use; if not specified, the default model is used')
    seg_args.add_argument('--gpu', default=True, type=bool, help='use GPU')
    seg_args.add_argument('--diameter', default=None, type=float, help='Mean grain diameter in pixels to rescale images to; default is None, which leads to automated size estimation')
    seg_args.add_argument('--min_size', default=0, type=float, help='Minimum object diameter in pixels to segement; default is 15 pixels')
    seg_args.add_argument('--skip_segmentation', default=False, type=bool, help='Skip segmentation and only calculate grain size distributions for already existing masks.')

    gs_args=parser.add_argument_group('Grain size estimation')
    gs_args.add_argument('--min_grain_size', type=float, default=None, help='Minimum grain size in pixels to consider for grain size estimation (default: None); grains with a fitted ellipse smaller than this size will be ignored.')
    gs_args.add_argument('--edge_filter', type=float, default=None, help = 'Edge filter to remove grains close to the image boundary (default: None).')
    gs_args.add_argument('--fit', type=str, default=None, help='Additional approximation for grains (default: None); options are convex hull (convh) or outline (outl).')
    gs_args.add_argument('--grid_resample', type=int, default=None, help = 'Resample images with a grid with a given resolution in pixel (default: None). Equivalent ot a digital Wolman grid.')
    gs_args.add_argument('--random_resample', type=int, default=None, help = 'Resample image with a random number of points (default: None).')
    gs_args.add_argument('--resolution', default=None, help = 'Image resolution to scale grain sizes to in mm/px (default: None). If a value is provided, the grain sizes will be scaled to the given resolution. Alternatively, can provided as path to a csv file with image_specific resolutions (see template). For estimating the image resolution from camera parameters see the preprocessing notebook.')

    args = parser.parse_args()

    mute = False if args.mute_output else True

    if args.img_dir == None or os.path.exists(args.img_dir) == False:
        print('>> Please specify a valid input directory for images to segment.')
        exit()
    
    skip_segmentation = True if args.skip_segmentation else False
    if mute == False and skip_segmentation == True:
        print('>> Skipping segmentation and only measuring grains for already existing masks.')

    if args.model_dir == None and skip_segmentation==False:
        if mute == False:
            print('>> No model specified. Using default model.')
        parent = str(Path(os.getcwd()).parent)
        args.model_dir = parent+'/models/full_set_1.170223'
        if os.path.exists(args.model_dir) == False:
            print('>> Default model not found. Please provide a valid model path or re-download the default models.')
            exit()
    
    TAR_DIR = '' if args.out_dir == None else args.out_dir

    #segmentation
    if skip_segmentation == False:
        segmentation_step(args,mute=mute,TAR_DIR=TAR_DIR)
    
    #grain size estimation

    #set filters
    filters = filters= {'edge':[False,.1],'px_cutoff':[False,12]}
    if args.min_grain_size:
        filters['px_cutoff'] = [True,args.min_grain_size]
    if args.edge_filter:
        filters['edge'] = [True,args.edge_filter]
    print(f'>> ImageGrains: Measuring grains for masks in {args.img_dir}.')

    #optional resampling (if done, sub-directory with resampled masks will be created)
    resampled = None
    if args.grid_resample or args.random_resample:
        resample_path = resampling_step(args,filters,mute=mute,TAR_DIR=TAR_DIR)
        resampled = True 
    
    #add additional approximation (convh, outl)
    fit_method = args.fit
    grain_size_step(args.img_dir,filters,fit_method=fit_method,mute=mute,TAR_DIR=TAR_DIR)    
    if resampled == True:
        grain_size_step(resample_path,filters,fit_method=fit_method,mute=True,TAR_DIR=TAR_DIR)
    
    #optional scaling of grains
    if args.resolution:
        scaling_step(args.img_dir,args.resolution,mute=mute,TAR_DIR=TAR_DIR)
        if resampled == True:
            scaling_step(resample_path,args.resolution,gsd_str='resampled_grains',mute=True,TAR_DIR=TAR_DIR)

def segmentation_step(args,mute=False,TAR_DIR=''):
    if args.gpu == True:
        if torch.cuda.is_available() == True and mute == False:
            print('>> Using GPU: ',torch.cuda.get_device_name(0))
        elif torch.cuda.is_available() == False and mute== False:
            print('>> GPU not available - check if correct pytorch version is installed. Using CPU instead.')

    print('>> ImageGrains: Segmenting ',args.img_type,' images in ',args.img_dir)
    _ = segmentation_helper.batch_predict(args.model_dir,args.img_dir,TAR_DIR=TAR_DIR,
                                    image_format=args.img_type,use_GPU=args.gpu,diameter=args.diameter, min_size=args.min_size,
                                        mute=mute,return_results=False,save_masks=True)

    if '.' in args.model_dir:
        M_ID = [Path(args.model_dir).stem]
    else:
        _,M_ID = segmentation_helper.models_from_zoo(args.model_dir)

    #segmentation example plot
    if mute == False:
        for ID in M_ID:
            imgs,_,preds = data_loader.dataset_loader(args.img_dir,pred_str=f'{ID}')
            if len(imgs) > 6:
                rng = default_rng()
                numbers = rng.choice(len(imgs), size=6, replace=False)
                imgs = [imgs[x] for x in numbers]
                preds = [preds[x] for x in numbers]
            pred_plot = plotting.inspect_predictions(imgs,preds,title=f"Segmentation examples for {ID}")
            if TAR_DIR != '':
                out_dir = TAR_DIR
            else:
                out_dir = args.img_dir
            pred_plot.savefig(out_dir+f'/{ID}_prediction_examples.png',dpi=300)
    return

def resampling_step(args,filters,mute=False,TAR_DIR=''):    
    if args.grid_resample:
        method = 'wolman'
        grid_size= args.grid_resample
        if mute == False:
            print('>> Resampling grains with a grid with a resolution of ',args.grid_resample,' pixels.')

    elif args.random_resample:
        method = 'random'
        n_rand = args.random_resample
        if mute == False:
            print('>> Resampling grains with a random number of points with a maximum of ',args.random_resample,' points.')
    _,_,masks= data_loader.dataset_loader(args.img_dir)
    
    for _,mask in enumerate(masks):
        #get ID from file name
        maskID = Path(mask).stem
        if 'flow' in maskID: #catch flow representations from potentially present from training
            continue
        else:
            #load masks from file
            mask = io.imread(mask)
            #resample mask to grid
            if method == 'wolman':
                grid_resampled,_,_ = grainsizing.resample_masks(mask,filters=filters,method = method, grid_size=grid_size,mute=True)
            else:
                grid_resampled,_,_ = grainsizing.resample_masks(mask,filters=filters,mute=True,method=method,n_rand=n_rand)
            #save resampled mask to file
            if not TAR_DIR:
                resampled_dir = args.img_dir+'/resampled/' 
                os.makedirs(resampled_dir, exist_ok=True)
                io.imsave(resampled_dir + maskID +f'_{method}_resampled.tif',grid_resampled)
            else:
                resampled_dir = TAR_DIR +'/resampled/' 
                os.makedirs(resampled_dir, exist_ok=True)
                io.imsave(resampled_dir + maskID+f'_{method}_resampled.tif',grid_resampled)
    return resampled_dir

def grain_size_step(img_dir,filters,fit_method=None,mute=False,TAR_DIR=''):
    if not fit_method:
        _,_,_ = grainsizing.batch_grainsize(img_dir,filters=filters,mute=True,TAR_DIR=TAR_DIR)
    else:
        if mute== False:
            print('>> Adding additional approximation for grains: ',fit_method)
        _,_,_ = grainsizing.batch_grainsize(img_dir,filters=filters,fit_method=fit_method,mute=True,TAR_DIR=TAR_DIR)
    return

def scaling_step(img_dir,resolution,mute=False,gsd_str='_grains',TAR_DIR=''):
    try:
        res = float(resolution)
        if type(res) == float:
            _ = grainsizing.re_scale_dataset(img_dir,resolution=res,gsd_str=gsd_str,save_gsds=True, TAR_DIR=TAR_DIR)
        if mute == False:
                print('>> Scaled grains with a spacing of',res,'mm/px.')
    except ValueError:
        pass
    if os.path.exists(resolution) == True:        
        #load names and resolutions
        df = pd.read_csv(resolution)
        names = df['name']
        resolutions = df['resolution']
        #load grains from files
        grains = data_loader.load_grain_set(img_dir)
        #find matching names
        new_res = []
        for kk in range(len(grains)):
            gr_ID = Path(grains[kk]).stem
            for namei, resi in zip(names, resolutions):
                if namei in gr_ID:
                    new_res.append(resi)    
        _ = grainsizing.re_scale_dataset(img_dir,resolution=new_res,gsd_str=gsd_str,save_gsds=True, TAR_DIR=TAR_DIR)
        if mute == False:
            print('>> Rescaled grains image-specific resolutions from',resolution,'.')
        return

if __name__ == '__main__':
    main()
