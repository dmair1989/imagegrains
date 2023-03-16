import os,pickle
import pandas as pd
from pathlib import Path
from glob import glob
from natsort import natsorted

def find_data(PATH,mask_str='mask',im_str='',im_format='jpg',mask_format='tif'):
    """
    This function loads images and masks from a directory. The directory can contain a 'train' and 'test' subfolder.

    Parameters:
    ------------
    PATH (str) - Path to the directory containing the images and masks
    mask_str (str(optional, default 'mask')) - A string used to filter the masks in `PATH`
    im_str (str(optional, default '')) - A string used to filter the images in `PATH`
    im_format (str(optional, default 'jpg')) - Image format of the images in `PATH`
    mask_format (str(optional, default 'tif')) - Image format of the masks in `PATH`

    Returns:
    ------------
    train_images (list) - List of paths to the training images
    train_masks (list) - List of paths to the training masks
    test_images (list) - List of paths to the test images
    test_masks (list) - List of paths to the test masks

    """
    try:
        dirs = next(os.walk(PATH))[1]
    except StopIteration:
        dirs=[]
    W_PATH = []
    if not dirs:
        W_PATH = PATH
        train_images = find_imgs_masks(W_PATH,format=im_format,filter_str=im_str)
        train_masks = find_imgs_masks(W_PATH,format=mask_format,filter_str=mask_str)
        test_images,test_masks = [],[]
    else:
        for dir in dirs:
            if 'test' in dir:
                W_PATH = str(PATH+'/test/')
                test_images = find_imgs_masks(W_PATH,format=im_format,filter_str=im_str)
                test_masks = find_imgs_masks(W_PATH,format=mask_format,filter_str=mask_str)
            if 'train' in dir:
                W_PATH = str(PATH+'/train/')
                train_images = find_imgs_masks(W_PATH,format=im_format,filter_str=im_str)
                train_masks = find_imgs_masks(W_PATH,format=mask_format,filter_str=mask_str)
    return train_images,train_masks,test_images,test_masks

def find_imgs_masks(W_PATH,format='',filter_str=''):
    """
    This function loads images and masks from a directory.

    Parameters:
    ------------
    W_PATH (str) - Path to the directory containing the images and masks
    format (str) - Image format of the images in `W_PATH`
    filter_str (str) - A string used to filter the images in `W_PATH`

    Returns:
    ------------
    ret_list (list) - List of paths to the images

    """
    ret_list = natsorted(glob(W_PATH+'/*'+filter_str+'*.'+format))
    return ret_list

def dataset_loader(IM_DIRs,image_format='jpg',label_format='tif',pred_format='tif',label_str='',pred_str=''):
    """
    Loads images, labels, and predictions from a folder or list of folders.

    Parameters:
    ------------
    IM_DIRs (str or list of str) - image directory or list of image directories.
    image_format (str (optional, default='jpg')) - The file format of the images.
    label_format (str (optional, default='tif')) - The file format of the labels.
    pred_format (str (optional, default='tif')) - The file format of the predictions.
    label_str (str (optional, default='')) - A string to search for in label file name.
    pred_str (str (optional, default='')) - A string to search for in prediction file name.

    Returns
    ------------
    imgs (list) - list of images
    lbls (list) - list of labels
    preds (list) - list of predictions      
    
    """
    imgs,lbls,preds = [],[],[]
    if type(IM_DIRs) == list:
        dirs = []
        for x in range(len(IM_DIRs)):
            try:
                dirs += next(os.walk(IM_DIRs[x]))[1]
            except StopIteration:
                continue
    else:
        try:
            dirs = next(os.walk(IM_DIRs))[1]
        except StopIteration:
            dirs=[]
    IM_DIR = []
    if dirs:
        for dir in dirs:
            if 'test' in dir:
                IM_DIR += [str(IM_DIRs+'/test/')]
            if 'train' in dir:
                IM_DIR += [str(IM_DIRs+'/train/')]
        for dir in IM_DIR:
            imgs1,lbls1,preds1 = load_from_folders(dir,image_format=image_format,label_format=label_format,pred_format=pred_format,label_str=label_str,pred_str=pred_str)
            imgs += imgs1
            lbls += lbls1
            preds += preds1
    if not IM_DIR:
        IM_DIR = IM_DIRs
        imgs1,lbls1,preds1 = load_from_folders(IM_DIR,image_format=image_format,label_format=label_format,pred_format=pred_format,label_str=label_str,pred_str=pred_str)
        imgs += imgs1
        lbls += lbls1
        preds += preds1
    
    return imgs,lbls,preds

def load_from_folders(IM_DIR,LBL_DIR='',PRED_DIR='',image_format='jpg',label_format='tif',pred_format='tif',label_str='',pred_str=''):
    """
    Loads images, labels, and predictions from separate folders.

    Parameters:
    ------------
    IM_DIR (str) - image directory.
    LBL_DIR (str (optional, default='')) - label directory.
    PRED_DIR (str (optional, default='')) - prediction directory.
    image_format (str (optional, default='jpg')) - The file format of the images.
    label_format (str (optional, default='tif')) - The file format of the labels.
    pred_format (str (optional, default='tif')) - The file format of the predictions.
    label_str (str (optional, default='')) - A string to search for in label file name.
    pred_str (str (optional, default='')) - A string to search for in prediction file name.

    Returns
    ------------    
    imgs (list) - list of images
    lbls (list) - list of labels
    preds (list) - list of predictions

    """
    if LBL_DIR:
        lbls = natsorted(glob(LBL_DIR+'/*'+label_str+'*.'+label_format))
    else:
        lbls = natsorted(glob(IM_DIR+'/*'+label_str+'*.'+label_format))
    if PRED_DIR:
        preds = natsorted(glob(PRED_DIR+'/*'+pred_str+'*.'+pred_format))
    else:
        preds = natsorted(glob(IM_DIR+'/*'+pred_str+'*.'+pred_format))
    imgs = natsorted(glob(IM_DIR+'/*.'+image_format))
    if not any(imgs) and not any(preds) and not any(lbls):
        print('Could not load any images and/or masks.')
    return imgs,lbls,preds

def load_eval_res(name,PATH=''):
    """
    Loads evaluation results from a pkl file.
    
    Parameters:
    ------------
    name (str) - name of the pkl file
    PATH (str (optional, default='')) - Path to the pkl file
    
    Returns
    ------------
    eval_results (dict) - Dictionary of evaluation results

    """
    with open(PATH +'/'+ name +'.pkl', 'rb') as f:
        eval_results = pickle.load(f)
    return eval_results

def load_grain_set(DIR,gsd_format='csv',gsd_str='grains'):
        """
        Loads a grain size distributions from a directory.

        Parameters
        ----------
        DIR (str) - directory of the grain size distributions
        gsd_format (str (optional, default = 'csv')) - format of the grain size distributions
        gsd_str (str (optional, default = 'grains')) - string to filter the grain size distributions

        Returns
        -------
        gsds (list) - list of grain size distributions
                
        """
        if type(DIR) == list:
            dirs = []
            for x in range(len(DIR)):
                try:
                    dirs += next(os.walk(DIR[x]))[1]
                except StopIteration:
                    continue
        else:
            try:
                dirs = next(os.walk(DIR))[1]
            except StopIteration:
                dirs=[]
        G_DIR = []
        if dirs:
            gsds=[]
            for dir in dirs:
                if 'test' in dir:
                        G_DIR += [str(DIR+'/test/')]
                if 'train' in dir:
                        G_DIR += [str(DIR+'/train/')]
            for path in G_DIR:
                gsds += gsds_from_folder(path,gsd_format=gsd_format,gsd_str=gsd_str) 
        if not G_DIR:
            gsds=[]
            G_DIR = [DIR]
            for path in G_DIR:
                gsds += gsds_from_folder(path,gsd_format=gsd_format,gsd_str=gsd_str)
        return gsds

def read_grains(PATH,sep=',',column_name='ell: b-axis (mm)'):
    df = pd.read_csv(PATH,sep=sep)
    grains = df[column_name].values
    return grains
    
def gsds_from_folder(PATH,gsd_format='csv',gsd_str='grains'):
    gsds_raw = natsorted(glob(PATH+'/*'+gsd_str+'*.'+gsd_format))
    gsds = []
    [gsds.append(gsd) for gsd in gsds_raw]
    return gsds

def read_set_unc(PATH,mc_str='uncert'):
    """
    Returns a filtered list of all uncertainty files and a list of all IDs.
    """
    try:
        dirs = next(os.walk(PATH))[1]
    except StopIteration:
        dirs = []
    G_DIR = []
    if dirs:
        if 'test' in dirs:
            G_DIR = [str(PATH+'/test/')]
        if 'train' in dirs:
            G_DIR += [str(PATH+'/train/')]
    if not G_DIR:
        G_DIR = [PATH]
    mcs,ids=[],[]
    for path in G_DIR:
        mc= natsorted(glob(path+'/*'+mc_str+'*.txt'))
        im= natsorted(glob(path+'/*'+'*.jpg'))
        id_i = [Path(im[idx]).stem for idx in range(len(im))]
        #id_i = [im[i].split('\\')[1].split('.')[0] for i in range(len(im))]
        mcs+=mc
        ids+=id_i
    return mcs,ids

def read_unc(path,sep=','):
    """
    Reads uncertainty file and returns a dataframe.
    """
    df = pd.read_csv(path,sep=sep, header=None)
    df = df.T
    df.columns = ['data','med','uci','lci']
    df = df.round(decimals=2)
    return df