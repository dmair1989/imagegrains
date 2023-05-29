import os,pickle
import pandas as pd
from pathlib import Path
from glob import glob
from natsort import natsorted

def find_data(image_path,mask_str='mask',im_str='',im_format='jpg',mask_format='tif'):
    """
    This function loads images and masks from a directory. The directory can contain a 'train' and 'test' subfolder.

    Parameters:
    ------------
    image_path (str) - Path to the directory containing the images and masks
    mask_str (str(optional, default 'mask')) - A string used to filter the masks in `image_path`
    im_str (str(optional, default '')) - A string used to filter the images in `image_path`
    im_format (str(optional, default 'jpg')) - Image format of the images in `image_path`
    mask_format (str(optional, default 'tif')) - Image format of the masks in `image_path`

    Returns:
    ------------
    train_images (list) - List of paths to the training images
    train_masks (list) - List of paths to the training masks
    test_images (list) - List of paths to the test images
    test_masks (list) - List of paths to the test masks

    """
    try:
        dirs = next(os.walk(image_path))[1]
    except StopIteration:
        dirs=[]
    working_directory = []
    if not dirs:
        working_directory = image_path
        train_images = find_imgs_masks(working_directory,format=im_format,filter_str=im_str)
        train_masks = find_imgs_masks(working_directory,format=mask_format,filter_str=mask_str)
        test_images,test_masks = [],[]
    else:
        for dir in dirs:
            if 'test' in dir:
                working_directory = str(image_path+'/test/')
                test_images = find_imgs_masks(working_directory,format=im_format,filter_str=im_str)
                test_masks = find_imgs_masks(working_directory,format=mask_format,filter_str=mask_str)
            if 'train' in dir:
                working_directory = str(image_path+'/train/')
                train_images = find_imgs_masks(working_directory,format=im_format,filter_str=im_str)
                train_masks = find_imgs_masks(working_directory,format=mask_format,filter_str=mask_str)
    return train_images,train_masks,test_images,test_masks

def find_imgs_masks(image_path,format='',filter_str=''):
    """
    This function loads images and masks from a directory.

    Parameters:
    ------------
    image_path (str) - Path to the directory containing the images and masks
    format (str) - Image format of the images in `image_path`
    filter_str (str) - A string used to filter the images in `image_path`

    Returns:
    ------------
    ret_list (list) - List of paths to the images

    """
    ret_list = natsorted(glob(image_path+'/*'+filter_str+'*.'+format))
    return ret_list

def dataset_loader(image_directories,image_format='jpg',label_format='tif',pred_format='tif',label_str='',pred_str=''):
    """
    Loads images, labels, and predictions from a folder or list of folders.

    Parameters:
    ------------
    image_directories (str or list of str) - image directory or list of image directories.
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
    if type(image_directories) == list:
        dirs = []
        for x in range(len(image_directories)):
            try:
                dirs += next(os.walk(image_directories[x]))[1]
            except StopIteration:
                continue
    else:
        try:
            dirs = next(os.walk(image_directories))[1]
        except StopIteration:
            dirs=[]
    image_directory = []
    if dirs:
        for dir in dirs:
            if 'test' in dir:
                image_directory += [str(image_directories+'/test/')]
            if 'train' in dir:
                image_directory += [str(image_directories+'/train/')]
        for dir in image_directory:
            imgs1,lbls1,preds1 = load_from_folders(dir,image_format=image_format,label_format=label_format,pred_format=pred_format,label_str=label_str,pred_str=pred_str)
            imgs += imgs1
            lbls += lbls1
            preds += preds1
    if not image_directory:
        image_directory = image_directories
        imgs1,lbls1,preds1 = load_from_folders(image_directory,image_format=image_format,label_format=label_format,pred_format=pred_format,label_str=label_str,pred_str=pred_str)
        imgs += imgs1
        lbls += lbls1
        preds += preds1
    
    return imgs,lbls,preds

def load_from_folders(image_directory,label_directory='',pred_directory='',image_format='jpg',label_format='tif',pred_format='tif',label_str='',pred_str=''):
    """
    Loads images, labels, and predictions from separate folders.

    Parameters:
    ------------
    image_directory (str) - image directory.
    label_directory (str (optional, default='')) - label directory.
    pred_directory (str (optional, default='')) - prediction directory.
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
    if label_directory:
        lbls = natsorted(glob(label_directory+'/*'+label_str+'*.'+label_format))
    else:
        lbls = natsorted(glob(image_directory+'/*'+label_str+'*.'+label_format))
    if pred_directory:
        preds = natsorted(glob(pred_directory+'/*'+pred_str+'*.'+pred_format))
    else:
        preds = natsorted(glob(image_directory+'/*'+pred_str+'*.'+pred_format))
    imgs = natsorted(glob(image_directory+'/*.'+image_format))
    if not any(imgs) and not any(preds) and not any(lbls):
        print('Could not load any images and/or masks.')
    return imgs,lbls,preds

def load_eval_res(name,file_path=''):
    """
    Loads evaluation results from a pkl file.
    
    Parameters:
    ------------
    name (str) - name of the pkl file
    file_path (str (optional, default='')) - Path to the pkl file
    
    Returns
    ------------
    eval_results (dict) - Dictionary of evaluation results

    """
    with open(file_path +'/'+ name +'.pkl', 'rb') as f:
        eval_results = pickle.load(f)
    return eval_results

def load_grain_set(file_dir,gsd_format='csv',gsd_str='grains'):
        """
        Loads a grain size distributions from a directory.

        Parameters
        ----------
        file_dir (str) - directory of the grain size distributions
        gsd_format (str (optional, default = 'csv')) - format of the grain size distributions
        gsd_str (str (optional, default = 'grains')) - string to filter the grain size distributions

        Returns
        -------
        gsds (list) - list of grain size distributions
                
        """
        if type(file_dir) == list:
            dirs = []
            for x in range(len(file_dir)):
                try:
                    dirs += next(os.walk(file_dir[x]))[1]
                except StopIteration:
                    continue
        else:
            try:
                dirs = next(os.walk(file_dir))[1]
            except StopIteration:
                dirs=[]
        active_file_dir = []
        if dirs:
            gsds=[]
            for dir in dirs:
                if 'test' in dir:
                        active_file_dir += [str(file_dir+'/test/')]
                if 'train' in dir:
                        active_file_dir += [str(file_dir+'/train/')]
            for path in active_file_dir:
                gsds += gsds_from_folder(path,gsd_format=gsd_format,gsd_str=gsd_str) 
        if not active_file_dir:
            gsds=[]
            active_file_dir = [file_dir]
            for path in active_file_dir:
                gsds += gsds_from_folder(path,gsd_format=gsd_format,gsd_str=gsd_str)
        return gsds

def read_grains(file_path,sep=',',column_name='ell: b-axis (px)'):
    df = pd.read_csv(file_path,sep=sep)
    grains = df[column_name].values
    return grains
    
def gsds_from_folder(file_path,gsd_format='csv',gsd_str='grains'):
    gsds_raw = natsorted(glob(file_path+'/*'+gsd_str+'*.'+gsd_format))
    gsds = []
    [gsds.append(gsd) for gsd in gsds_raw]
    return gsds

def read_set_unc(file_path,unc_str='_perc_uncert',file_format='txt'):
    """
    Returns a filtered list of all uncertainty files and a list of all IDs.
    """
    try:
        dirs = next(os.walk(file_path))[1]
    except StopIteration:
        dirs = []
    active_file_dir = []
    if dirs:
        if 'test' in dirs:
            active_file_dir = [str(file_path+'/test/')]
        if 'train' in dirs:
            active_file_dir += [str(file_path+'/train/')]
    if not active_file_dir:
        active_file_dir = [file_path]
    mcs,ids=[],[]
    for path in active_file_dir:
        mc= natsorted(glob(path+'/*'+unc_str+'*.'+file_format))
        id_i = [Path(mc[idx]).stem for idx in range(len(mc))]
        mcs+=mc
        ids+=id_i
    return mcs,ids

def read_unc(path,sep=',',file_format='txt'):
    """
    Reads uncertainty file and returns a dataframe.
    """
    df = pd.read_csv(path,sep=sep, header=None)
    if file_format == 'txt':
        df = df.T
    df.columns = ['data','med','uci','lci']
    df = df.round(decimals=2)
    return df

def get_img_name_for_summary(file_path, imgs = None, p_string= '_full',overwrite = True):
    df = pd.read_csv(file_path)
    imgs_stem = [Path(img).stem for img in imgs]
    imgs_name = [Path(img).name for img in imgs]
    for i in range(len(df)):
        a=Path(df['Image/Masks'][i]).stem.split(p_string)[0]
        for j,b in enumerate(imgs_stem):
            if a==b:
                df.at[i,'Image_Name'] = imgs_name[j]
    if overwrite == True:
        df.to_csv(file_path, index = False)
    return df