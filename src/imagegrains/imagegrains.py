import os, argparse, glob, pathlib, time
import numpy as np
from cellpose import models, utils, io
from imagegrains import import segmentation_helper
from imagegrains import import grainsizing
from imagegrains import import data_loader
from imagegrains import import gsd_uncertainty
from imagegrains import import plotting

#will be __main__.py in the future
def main():
    parser = argparse.ArgumentParser(description='ImageGrains')
    parser.add_argument('--mute', default=True, type=bool, help='Mute console output')

    seg_args=parser.add_argument_group('Segmentation')
    seg_args.add_argument('--img_dir', type=str, help='Input directory for images to segment')
    seg_args.add_argument('--out_dir', default=[], type=str, help='Output directory for segmented images')
    seg_args.add_argument('--img_type', default='jpg', type=str, help='Image type to segment')
    seg_args.add_argument('--model_dir', default='models/full_set_1.170223', type=str, help='Segemntation model to use')
    seg_args.add_argument('--gpu', default=True, type=bool, help='use GPU')
    seg_args.add_argument('--diameter', default=0, type=int, help='Mean grain diameter in pixels to rescale images to')

    args = parser.parse_args()
    print('>> ImageGrains: Segmenting ',args.img_type,' files in ',args.img_dir)
    _ = segmentation_helper.batch_predict(args.model_dir,args.img_dir,mute=args.mute,return_results=False,save_masks=True)

if __name__ == '__main__':
    main()
