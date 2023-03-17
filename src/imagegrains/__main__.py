import os, argparse
from pathlib import Path
import torch
import pandas as pd
from numpy.random import default_rng
from cellpose import io
from imagegrains import segmentation_helper, grainsizing, gsd_uncertainty
from imagegrains import data_loader, plotting

def main():
    parser = argparse.ArgumentParser(description='ImageGrains')
    parser.add_argument('--img_dir', default=None, type=str, help='Input directory for images to segment')
    parser.add_argument('--skip_plots', default=False, type=bool, help='Skip the overview plots')

    seg_args=parser.add_argument_group('Segmentation')
    seg_args.add_argument('--mute_output', default=None, type=bool, help='Mute console output. If True, example plots will not be saved.')
    seg_args.add_argument('--out_dir', default=None, type=str, help='Output directory for segmented images; if not specified, the images will be saved in the same directory as the input images')
    seg_args.add_argument('--img_type', default='jpg', type=str, help='Image type to segment; by default the the script will look for .jpg files. Alternatively, .tif and .png files can be segmented.')
    seg_args.add_argument('--model_dir', default=None, type=str, help='Segemntation model to use; if not specified, the default model is used')
    seg_args.add_argument('--gpu', default=True, type=bool, help='use GPU')
    seg_args.add_argument('--diameter', default=None, type=float, help='Mean grain diameter in pixels to rescale images to; default is None, which leads to automated size estimation')
    seg_args.add_argument('--min_size', default=0, type=float, help='Minimum object diameter in pixels to segement; default is 15 pixels')
    seg_args.add_argument('--skip_segmentation', default=False, type=bool, help='Skip segmentation and only calculate grain size distributions for already existing masks.')

    gs_args=parser.add_argument_group('Grain size estimation')
    gs_args.add_argument('--filter_str', type=str, default=None, help='Filter mask files with optional strin (default: None.')
    gs_args.add_argument('--min_grain_size', type=float, default=None, help='Minimum grain size in pixels to consider for grain size estimation (default: None); grains with a fitted ellipse smaller than this size will be ignored.')
    gs_args.add_argument('--edge_filter', type=float, default=None, help = 'Edge filter to remove grains close to the image boundary (default: None).')
    gs_args.add_argument('--switch_filters_off', type=bool, default=False, help = 'Switch off all filters fro grain sizing (default: False).')
    gs_args.add_argument('--fit', type=str, default=None, help='Additional approximation for grains (default: None); options are convex hull (convex_hull) or outline (mask_outline).')
    gs_args.add_argument('--grid_resample', default=None, help = 'Resample images with a grid with a given resolution in pixel (default: None). Equivalent ot a digital Wolman grid.')
    gs_args.add_argument('--random_resample', default=None, help = 'Resample image with a random number of points (default: None).')
    gs_args.add_argument('--resolution', default=None, help = 'Image resolution to scale grain sizes to in mm/px (default: None). If a value is provided, the grain sizes will be scaled to the given resolution. Alternatively, can provided as path to a csv file with image_specific resolutions (see template). For estimating the image resolution from camera parameters see the preprocessing notebook.')

    gsd_args=parser.add_argument_group('GSD analysis')
    gsd_args.add_argument('--unc_method', type = str, default = 'bootstrapping', help = 'Method to estimate uncertainty of grain size distribution (default: bootstrap). Options are bootstraping (bootstrapping), simpple Monte Carlo (MC), or advanced Monte Carlo for SfM data (MC_SfM_OM or MC_SfM_SI).')
    gsd_args.add_argument('--n', type = int, default = 1000, help = 'Number of iterations for uncertainty estimation (default: 10000).')
    gsd_args.add_argument('--scale_err', default=0.1, help='Scale error for MC uncertainty estimation in fractions (default: 0.1).')
    gsd_args.add_argument('--length_err', default=0.1, help='Length error for MC uncertainty estimation in pixel or mm (default: 1); whether it is interpreted as py or mm value depends resolution was provided or not.')
    gsd_args.add_argument('--SfM_file', default = None, help = 'Path to SfM uncertainty file (default: None). See template for details.')
    gsd_args.add_argument('--summary', type = bool, default = True, help = 'Save key percentile summary for all images (default: True).')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit()

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
    filters = filters= {'edge':[True,.1],'px_cutoff':[True,12]}
    if args.min_grain_size:
        filters['px_cutoff'] = [True,args.min_grain_size]
    if args.edge_filter:
        filters['edge'] = [True,args.edge_filter]
    if args.switch_filters_off:
        filters= {'edge':[False,.1],'px_cutoff':[False,12]}
    if mute == False:
        print(f'Filter configuratiuon: {filters}')

    print(f'>> ImageGrains: Measuring grains for masks in {args.img_dir}.')

    #optional resampling (if done, sub-directory with resampled masks will be created)
    resampled = None
    if args.grid_resample or args.random_resample:
        resample_path = resampling_step(args,filters,mute=mute,TAR_DIR=TAR_DIR)
        resampled = True 
    
    #add additional approximation (convh, outl)
    fit_method = args.fit
    mask_str = args.filter_str if args.filter_str else ''
    grain_size_step(args.img_dir,filters,fit_method=fit_method,mute=mute,TAR_DIR=TAR_DIR,mask_str=mask_str)    
    if resampled == True:
        grain_size_step(resample_path,filters,fit_method=fit_method,mute=True,TAR_DIR=TAR_DIR,mask_str=mask_str)
    
    #optional scaling of grains
    if args.resolution:
        scaling_step(args.img_dir,args.resolution,mute=mute,TAR_DIR=TAR_DIR)
        if resampled == True:
            scaling_step(resample_path,args.resolution,gsd_str='resampled_grains',mute=True,TAR_DIR=TAR_DIR)

    #gsd analysis
    print ('>> ImageGrains: Calculating grain size distributions and uncertainties for ',args.img_dir)
    gsd_step(args.img_dir,args,mute=mute,TAR_DIR=TAR_DIR)
    if resampled == True:
        print('>> Calculating grain size distributions and uncertainties for ',resample_path)
        gsd_step(resample_path,args,mute=mute,TAR_DIR=TAR_DIR)


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
    if not args.skip_plots:
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
        grid_size= int(args.grid_resample)
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
                resampled_dir = args.img_dir+'/Resampled_grains/' 
                os.makedirs(resampled_dir, exist_ok=True)
                io.imsave(resampled_dir + maskID +f'_{method}_resampled.tif',grid_resampled)
            else:
                resampled_dir = TAR_DIR +'/Resampled_grains/' 
                os.makedirs(resampled_dir, exist_ok=True)
                io.imsave(resampled_dir + maskID+f'_{method}_resampled.tif',grid_resampled)
    return resampled_dir

def grain_size_step(img_dir,filters,fit_method=None,mute=False,TAR_DIR='',mask_str=''):
    if not fit_method:
        _,_,_ = grainsizing.batch_grainsize(img_dir,filters=filters,mute=True,TAR_DIR=TAR_DIR,mask_str=mask_str)
    else:
        if mute== False:
            print('>> Adding additional approximation for grains: ',fit_method)
        _,_,_ = grainsizing.batch_grainsize(img_dir,filters=filters,fit_method=fit_method,mute=True,TAR_DIR=TAR_DIR,mask_str=mask_str)
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

def gsd_step(PATH,args,mute=False,TAR_DIR=''):
    grains= data_loader.load_grain_set(PATH,gsd_str='grains_re_scaled')
    scaled = True
    if not grains:
        grains = data_loader.load_grain_set(PATH,gsd_str='grains')
        scaled = False
        if mute == False:
            print('No scaled grains found. Loading unscaled grains.')
    #configure columns, uncertainty method and input for uncertainty estimation
    if scaled == True:
        columns = ['ell: a-axis (mm)','ell: b-axis (mm)']
        if args.fit == 'convex_hull':
            columns += ['convex hull: a axis (mm)','convex hull: b axis (mm)']
        if args.fit == 'mask_outline':
            columns += ['mask outline: a axis (mm)','mask outline: b axis (mm)']
        method = args.unc_method
        if method == 'MC_SfM':
            sfm_err = gsd_uncertainty.compile_sfm_error(args.SfM_file)
            if 'OM' in args.SfM_file:
                sfm_type = 'OM'
            else:
                sfm_type = 'SI'
        else: 
            sfm_err = None
            sfm_type = None
    else:
        columns = ['ell: a-axis (px)','ell: b-axis (px)']
        if args.fit == 'convex_hull':
            columns += ['convex hull: a axis (px)','convex hull: b axis (px)']
        if args.fit == 'mask_outline':
            columns += ['mask outline: a axis (px)','mask outline: b axis (px)']
        method = 'bootstrapping'
    #estimate uncertainty on a column-by-column basis
    ids = [str(Path(x).stem) for x in grains]
    df_list = []
    for i,column in enumerate(columns):
        if mute == False:
            print(column)
        #call uncertainty estimation function
        column_unc = gsd_uncertainty.dataset_uncertainty(gsds=grains,gsd_id=ids,return_results=True,
                                                save_results=False,method=method,column_name=column,
                                                num_it=args.n,scale_err=args.scale_err,length_err=args.length_err,
                                                sfm_error=sfm_err,sfm_type=sfm_type,mute=True)        
        #save results to dataframe and update it for each column
        for j,idi in enumerate(ids):
            if i == 0:
                df = pd.DataFrame({f'{column}_perc_lower_CI':column_unc[idi][2],
                                f'{column}_perc_median':column_unc[idi][0],
                                f'{column}_perc_upper_CI':column_unc[idi][1],
                                f'{column}_perc_value':column_unc[idi][3]})
                df_list.append(df)
            else:
                df_list[j]=pd.concat([df_list[j],pd.DataFrame({f'{column}_perc_lower_CI':column_unc[idi][2],
                                f'{column}_perc_median':column_unc[idi][0],
                                f'{column}_perc_upper_CI':column_unc[idi][1],
                                f'{column}_perc_value':column_unc[idi][3]})],axis=1)
        
        #call key percentile summary function and save results for each column

        axis = 'b_axis' if 'b-axis' in column else 'a_axis'
        approx = 'convex_hull' if 'convex hull' in column else 'mask_outline' if 'mask outline' in column else 'ellipse'
        unit = 'mm' if 'mm' in column else 'px'
        sum_df = grainsizing.summary_statistics(grains,ids,res_dict=column_unc, save_summary=False,
                                        method=method,approximation=approx,axis=axis,unit=unit,data_id='')
        if TAR_DIR != '':
            out_dir2 = TAR_DIR
        else:
            out_dir2 = PATH
        sum_df.to_csv(f'{out_dir2}/{axis}_{unit}_{approx}_{method}.csv',index=False)

    #save full GSD+uncertainty dataframe to csv for each grain set
    if TAR_DIR != '':
        out_dir = TAR_DIR+'/GSD_uncertainty/'
    else:
        out_dir = PATH + '/GSD_uncertainty/'
    os.makedirs(out_dir,exist_ok=True)
    for idi, dfi in zip(ids,df_list):
        idi = idi.replace('grains','')
        dfi = dfi.round(decimals=2)
        dfi.to_csv(f'{out_dir}/{idi}_{method}_full_uncertainty.csv')
    return 

if __name__ == '__main__':
    main()