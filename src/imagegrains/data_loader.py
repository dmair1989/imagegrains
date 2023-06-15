import os,pickle
import pandas as pd
from pathlib import Path
from glob import glob
from natsort import natsorted
import urllib.request

def download_files(tar_path = None):
    if not tar_path:
        homepath = Path.home().joinpath('imagegrains')
    else:
        homepath = Path(tar_path)

    nb_list = ['1_image_segmentation.ipynb',
                '2_grain_sizes.ipynb',
                '3_gsd_analysis.ipynb',
                '4_train_cellpose_model.ipynb',
                'complete_imagegrains_analysis.ipynb']
    fh_test_list = ['4_P1060348_3.jpg', '4_P1060348_3_mask.tif']
    fh_train_list = ['1_P1060330_1.jpg',
                    '1_P1060330_1_mask.tif',
                    '2_P1060338_0.jpg',
                    '2_P1060338_0_mask.tif',
                    '3_P1060343_3.jpg',
                    '3_P1060343_3_mask.tif',
                    '5_P1060351_2.jpg',
                    '5_P1060351_2_mask.tif',
                    '6_P1060355_0.jpg',
                    '6_P1060355_0_mask.tif',
                    '7_P1060359_3.jpg',
                    '7_P1060359_3_mask.tif']
    dem_dat_list = ['FH_resolutions.csv', 'OM_err.csv', 'SI_err.csv','K1/K1_C2_385.jpg', 'K1_field_measurement.csv']
    model_list = ['fh_boosted_1.170223', 'full_set_1.170223']

    os.makedirs(homepath, exist_ok=True)
    url = "https://raw.githubusercontent.com/dmair1989/imagegrains/main/"
    os.makedirs(homepath.joinpath('notebooks'), exist_ok=True)
    for nb in nb_list:
        try:
            urllib.request.urlretrieve(f'{url}/notebooks/{nb}', homepath.joinpath('notebooks',nb))
        except:
            continue
    os.makedirs(homepath.joinpath('demo_data','FH','test'), exist_ok=True)
    for file in fh_test_list:
        try:
            urllib.request.urlretrieve(f'{url}/demo_data/FH/test/{file}', homepath.joinpath('demo_data','FH','test',file))
        except:
            continue
    os.makedirs(homepath.joinpath('demo_data','FH','train'), exist_ok=True)
    for file in fh_train_list:
        try:
            urllib.request.urlretrieve(f'{url}/demo_data/FH/train/{file}', homepath.joinpath('demo_data','FH','train',file))
        except:
            continue
    os.makedirs(homepath.joinpath('demo_data','K1'), exist_ok=True)
    for file in dem_dat_list:
        try:
            urllib.request.urlretrieve(f'{url}/demo_data/{file}', homepath.joinpath('demo_data',file))
        except:
            continue
    os.makedirs(homepath.joinpath('models'), exist_ok=True)
    for model in model_list:
        try:
            urllib.request.urlretrieve(f'{url}/models/{model}', homepath.joinpath('models',model))
        except:
            continue
    return Path(homepath).as_posix()

def find_data(image_path,mask_str='mask',im_str='',im_format='jpg',mask_format='tif'):
    """
    This function loads images and masks from a directory. The directory can contain a 'train' and 'test' subfolder.

    Parameters:
    ------------
    image_path (str, Path) - Path to the directory containing the images and masks
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
        dirs = next(os.walk(Path(image_path)))[1]
    except StopIteration:
        dirs=[]
    working_directory = []
    if not dirs:
        working_directory = Path(image_path)
        train_images = find_imgs_masks(working_directory,format=im_format,filter_str=im_str)
        train_masks = find_imgs_masks(working_directory,format=mask_format,filter_str=mask_str)
        test_images,test_masks = [],[]
    else:
        for dir in dirs:
            if 'test' in dir:
                #working_directory = str(image_path+'/test/')
                working_directory = f'{Path(image_path)}/test/'
                test_images = find_imgs_masks(working_directory,format=im_format,filter_str=im_str)
                test_masks = find_imgs_masks(working_directory,format=mask_format,filter_str=mask_str)
            if 'train' in dir:
                #working_directory = str(image_path+'/train/')
                working_directory = f'{Path(image_path)}/train/'
                train_images = find_imgs_masks(working_directory,format=im_format,filter_str=im_str)
                train_masks = find_imgs_masks(working_directory,format=mask_format,filter_str=mask_str)
    return train_images,train_masks,test_images,test_masks

def find_imgs_masks(image_path,format='',filter_str=''):
    """
    This function loads images and masks from a directory.

    Parameters:
    ------------
    image_path (str, Path) - Path to the directory containing the images and masks
    format (str) - Image format of the images in `image_path`
    filter_str (str) - A string used to filter the images in `image_path`

    Returns:
    ------------
    ret_list (list) - List of paths to the images

    """
    ret_list = natsorted(glob(f'{Path(image_path)}/*{filter_str}*.{format}'))
    #ret_list = natsorted(glob(image_path+'/*'+filter_str+'*.'+format))
    return ret_list

def dataset_loader(image_directories,image_format='jpg',label_format='tif',pred_format='tif',label_str='',pred_str=''):
    """
    Loads images, labels, and predictions from a folder or list of folders.

    Parameters:
    ------------
    image_directories (str, Path or list of str,Path) - image directory or list of image directories.
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
                dirs += next(os.walk(Path(image_directories[x])))[1]
            except StopIteration:
                continue
    else:
        try:
            dirs = next(os.walk(Path(image_directories)))[1]
        except StopIteration:
            dirs=[]
    image_directory = []
    if dirs:
        for dir in dirs:
            if 'test' in dir:
                #image_directory += [str(image_directories+'/test/')]
                image_directory += [f'{Path(image_directories)}/test/']
            if 'train' in dir:
                #image_directory += [str(image_directories+'/train/')]
                image_directory += [f'{Path(image_directories)}/train/']
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
    image_directory (str,Path) - image directory.
    label_directory (str,Path (optional, default='')) - label directory.
    pred_directory (str,Path (optional, default='')) - prediction directory.
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
        #lbls = natsorted(glob(label_directory+'/*'+label_str+'*.'+label_format))
        lbls = natsorted(glob(f'{label_directory}/*{label_str}*.{label_format}'))
    else:
        #lbls = natsorted(glob(image_directory+'/*'+label_str+'*.'+label_format))
        lbls = natsorted(glob(f'{image_directory}/*{label_str}*.{label_format}'))
    if pred_directory:
        #preds = natsorted(glob(pred_directory+'/*'+pred_str+'*.'+pred_format))
        preds = natsorted(glob(f'{pred_directory}/*{pred_str}*.{pred_format}'))
    else:
        #preds = natsorted(glob(image_directory+'/*'+pred_str+'*.'+pred_format))
        preds = natsorted(glob(f'{image_directory}/*{pred_str}*.{pred_format}'))
    imgs = natsorted(glob(f'{image_directory}/*.{image_format}'))
    if not any(imgs) and not any(preds) and not any(lbls):
        print('Could not load any images and/or masks.')
    return imgs,lbls,preds

def load_eval_res(name,file_path=''):
    """
    Loads evaluation results from a pkl file.
    
    Parameters:
    ------------
    name (str) - name of the pkl file
    file_path (str, Path (optional, default='')) - Path to the pkl file
    
    Returns
    ------------
    eval_results (dict) - Dictionary of evaluation results

    """
    with open(f'{Path(file_path)}/{name}.pkl', 'rb') as f:
        eval_results = pickle.load(f)
    return eval_results

def load_grain_set(file_dir,gsd_format='csv',gsd_str='grains'):
        """
        Loads a grain size distributions from a directory.

        Parameters
        ----------
        file_dir (str, Path) - directory of the grain size distributions
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
                    dirs += next(os.walk(Path(file_dir[x])))[1]
                except StopIteration:
                    continue
        else:
            try:
                dirs = next(os.walk(Path(file_dir)))[1]
            except StopIteration:
                dirs=[]
        active_file_dir = []
        if dirs:
            gsds=[]
            for dir in dirs:
                if 'test' in dir:
                        #active_file_dir += [str(file_dir+'/test/')]
                        active_file_dir += [f'{Path(file_dir)}/test/']
                if 'train' in dir:
                        #active_file_dir += [str(file_dir+'/train/')]
                        active_file_dir += [f'{Path(file_dir)}/train/']
            for path in active_file_dir:
                gsds += gsds_from_folder(path,gsd_format=gsd_format,gsd_str=gsd_str) 
        if not active_file_dir:
            gsds=[]
            active_file_dir = [file_dir]
            for path in active_file_dir:
                gsds += gsds_from_folder(path,gsd_format=gsd_format,gsd_str=gsd_str)
        return gsds

def read_grains(file_path,sep=',',column_name='ell: b-axis (px)'):
    df = pd.read_csv(Path(file_path),sep=sep)
    grains = df[column_name].values
    return grains
    
def gsds_from_folder(file_path,gsd_format='csv',gsd_str='grains'):
    #gsds_raw = natsorted(glob(file_path+'/*'+gsd_str+'*.'+gsd_format))
    gsds_raw = natsorted(glob(f'{Path(file_path)}/*{gsd_str}*.{gsd_format}'))
    gsds = []
    [gsds.append(gsd) for gsd in gsds_raw]
    return gsds

def read_set_unc(file_path,unc_str='_perc_uncert',file_format='txt'):
    """
    Returns a filtered list of all uncertainty files and a list of all IDs.
    """
    try:
        dirs = next(os.walk(Path(file_path)))[1]
    except StopIteration:
        dirs = []
    active_file_dir = []
    if dirs:
        if 'test' in dirs:
            #active_file_dir = [str(file_path+'/test/')]
            active_file_dir = [f'{Path(file_path)}/test/']
        if 'train' in dirs:
            #active_file_dir += [str(file_path+'/train/')]
            active_file_dir = [f'{Path(file_path)}/train/']
    if not active_file_dir:
        active_file_dir = [Path(file_path)]
    mcs,ids=[],[]
    for path in active_file_dir:
        #mc= natsorted(glob(path+'/*'+unc_str+'*.'+file_format))
        mc= natsorted(glob(f'{Path(path)}/*{unc_str}*.{file_format}'))
        id_i = [Path(mc[idx]).stem for idx in range(len(mc))]
        mcs+=mc
        ids+=id_i
    return mcs,ids

def read_unc(path,sep=',',file_format='txt'):
    """
    Reads uncertainty file and returns a dataframe.
    """
    df = pd.read_csv(Path(path),sep=sep, header=None)
    if file_format == 'txt':
        df = df.T
    df.columns = ['data','med','uci','lci']
    df = df.round(decimals=2)
    return df

def get_img_name_for_summary(file_path, imgs = None, p_string= '_full',overwrite = True):
    df = pd.read_csv(Path(file_path))
    imgs_stem = [Path(img).stem for img in imgs]
    imgs_name = [Path(img).name for img in imgs]
    for i in range(len(df)):
        a=Path(df['Image/Masks'][i]).stem.split(p_string)[0]
        for j,b in enumerate(imgs_stem):
            if a==b:
                df.at[i,'Image_Name'] = imgs_name[j]
    if overwrite == True:
        df.to_csv(Path(file_path), index = False)
    return df