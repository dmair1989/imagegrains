import os, argparse, glob, time
from pathlib import Path
import torch
import numpy as np
from numpy.random import default_rng
from cellpose import models, utils, io
from imagegrains import segmentation_helper, grainsizing, gsd_uncertainty
from imagegrains import data_loader, plotting

def main():
    parser = argparse.ArgumentParser(description='ImageGrains')

    seg_args=parser.add_argument_group('Segmentation')
    seg_args.add_argument('--mute_output', default=False, type=bool, help='Mute console output')
    seg_args.add_argument('--img_dir', type=str, help='Input directory for images to segment')
    seg_args.add_argument('--out_dir', default=None, type=str, help='Output directory for segmented images; if not specified, the images will be saved in the same directory as the input images')
    seg_args.add_argument('--img_type', default='jpg', type=str, help='Image type to segment; by default the the script will look for .jpg files. Alternatively, .tif and .png files can be segmented.')
    seg_args.add_argument('--model_dir', default=None, type=str, help='Segemntation model to use; if not specified, the default model is used')
    seg_args.add_argument('--gpu', default=True, type=bool, help='use GPU')
    seg_args.add_argument('--diameter', default=None, type=float, help='Mean grain diameter in pixels to rescale images to; default is None, which leads to automated size estimation')
    seg_args.add_argument('--min_size', default=0, type=float, help='Minimum object diameter in pixels to segement; default is 15 pixels')
    seg_args.add_argument('--skip_segmentation', default=False, type=bool, help='Skip segmentation and only calculate grain size distributions for already existing masks.')

    gs_args=parser.add_argument_group('Grain size estimation')
    gs_args.add_argument('--filter_masks', type=bool, default=True)
    gs_args.add_argument('--filter_masks', type=bool, default=True)
    gs_args.add_argument('--min_grain_size', type=float, default=12)
    gs_args.add_argument('--edge_filter', type=float, default=0.1)
    gs_args.add_argument('--other_fit', type=str, default=None)

    args = parser.parse_args()
    
    if args.model_dir == None:
        parent = str(Path(os.getcwd()).parent)
        args.model_dir= parent+'/models/full_set_1.170223'
    
    TAR_DIR = '' if args.out_dir == None else args.out_dir

    #segmentation
    if not args.skip_segmentation == True:
        if args.gpu == True:
            if torch.cuda.is_available() == True and args.mute_output == False:
                print('>> Using GPU: ',torch.cuda.get_device_name(0))
            elif torch.cuda.is_available() == False and args.mute_output == False:
                print('>> GPU not available - check if correct pytorch version is installed. Using CPU instead.')

        print('>> ImageGrains: Segmenting ',args.img_type,' images in ',args.img_dir)
        _ = segmentation_helper.batch_predict(args.model_dir,args.img_dir,TAR_DIR=TAR_DIR,
                                        image_format=args.img_type,use_GPU=args.gpu,diameter=args.diameter, min_size=args.min_size,
                                          mute=args.mute_output,return_results=False,save_masks=True)
    
        if '.' in args.model_dir:
            model_list = [args.model_dir]
            M_ID = [Path(args.model_dir).stem]
        else:
            model_list,M_ID = segmentation_helper.models_from_zoo(args.model_dir)
    
        #segmentation example plot
        if args.mute_output == False:
            for ID in M_ID:
                imgs,_,preds = data_loader.dataset_loader(args.img_dir,label_str='mask',pred_str=f'{ID}')
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

    #grain size estimation
            
if __name__ == '__main__':
    main()
