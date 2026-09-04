import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle, cv2, shutil
import numpy as np
import pandas as pd

from pathlib import Path
from cellpose import io
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from skimage.measure import label, regionprops_table
from tifffile import imwrite

from imagegrains import grainsizing, data_loader, plotting, __cp_version__
from cellpose import metrics, models, io

def check_labels(labels,tar_dir='', lbl_str='_mask', mask_format='tif'):
    """
    This function checks if the labels are in the correct format. If not, it renames the labels to the correct format.
    The labels are renamed to the format: <image_ID><lbl_str>.<mask_format>
    
    Parameters:
    ------------
    labels (list) - List of paths to the labels
    tar_dir (str(optional, default '')) - Target directory to save the renamed labels. If not specified, the labels are saved in the same directory as the original labels.
    lbl_str (str(optional, default '_mask')) - String to be added to the image ID to form the label ID
    mask_format (str(optional, default 'tif')) - Image format of the labels

    Returns:
    ------------
    track_l (list) - List of image IDs for which the labels were renamed
    
    """

    if tar_dir:
        os.makedirs(Path(tar_dir), exist_ok=True)

    track_l = []

    for label in labels:
        if lbl_str in label:
            continue
        else:
            img= io.imread(str(label))
            img_id = Path(label).stem
            print(img_id)
            imgpath = Path(tar_dir)/f'{img_id}{lbl_str}.{mask_format}'
            io.imsave(str(imgpath),img)
            track_l.append(img_id)

    if len(track_l) == 0:
        print('No files renamed.')

    return track_l

def check_im_label_pairs(img_list, lbl_list):
    """
    This function checks if the images and labels are paired correctly. If not, it returns a list of images for which the labels are missing.	

    Parameters:
    ------------
    img_list (list) - List of paths to the images
    lbl_list (list) - List of paths to the labels
    
    Returns:
    ------------
    error_list (list) - List of paths to the images for which the labels are missing

    """

    error_list=[]

    for image in img_list:
        img_id = Path(image).stem
        
        if any(img_id in x for x in lbl_list):
            continue
        else:
            error_list.append(image)

    if len(error_list)==0:
        print('All images have labels.')

    return error_list

def custom_train(image_path, pretrained_model = None,datstring = None,
                lr = 0.2, nepochs = 1000,chan1 = 0, chan2= 0, gpu = True, batch_size = 8, use_bfloat16 = True,
                mask_filter = '_mask', rescale = False, save_each = False, return_model = False,
                save_every = 100,model_name = None, label_check = True,cp_version=__cp_version__):
    
    """
    This function trains a model on the images and labels in the specified directory. The images and labels should be in the same directory. The labels should be in the format: <image_ID>_mask.<mask_format>
    
    Parameters:
    ------------
    image_path (str, Path) - Path to the directory containing the images and labels
    pretrained_model (str(optional, default None)) - Path to the pretrained model. If not specified, the model is trained from scratch.
    datstring (str(optional, default None)) - String to be added to the model name. 
    return_model (bool(optional, default False)) - If True, the model is returned
    model_name (str(optional, default None)) - Name of the model.
    label_check (bool(optional, default True)) - If True, the labels are checked for the correct format and the images and labels are checked for correct pairing. If the labels are not in the correct format, they are renamed to the correct format. If the images and labels are not paired correctly, a list of images for which the labels are missing is returned.
    use_bfloat16 (bool(optional, default True)) - If True, the model is trained using float16, which is faster but limits the number of objects to <65,535 per mask. Set to False for using float32, which allows for more objects but is slower.

    more parameters:
    https://cellpose.readthedocs.io/en/latest/api.html#cellpose.models.CellposeModel.train 

    Returns:
    ------------
    model (CellposeModel, optional) - Trained model
    
    """

    _, _ = io.logger_setup()
    train_images,train_masks,test_images,test_masks = data_loader.find_data(image_path,mask_str=mask_filter)

    if label_check == True:
        check_labels(train_masks);
        check_labels(test_masks);
        check_im_label_pairs(train_images,train_masks);
        check_im_label_pairs(test_images,test_masks);

    train_data,train_labels,test_data,test_labels = [],[],[],[]
    for x1,y1 in zip(train_images,train_masks):
        train_data.append(io.imread(str(x1)))
        train_labels.append(io.imread(str(y1)))

    for x2,y2 in zip(test_images,test_masks):
        test_data.append(io.imread(str(x2)))
        test_labels.append(io.imread(str(y2)))
    
    if not model_name:
        model_name = 'new_model'

    if cp_version > 3:
        channels = None

        if not pretrained_model:
            model = models.CellposeModel(gpu=gpu,pretrained_model=None,use_bfloat16=use_bfloat16)
        else:
            model = models.CellposeModel(gpu=gpu,pretrained_model=pretrained_model,use_bfloat16=use_bfloat16)

    else:
        channels = [chan1,chan2]
        if not datstring:
            datstring = '000815'
            model_name = f'{model_name}.{datstring}'
        if pretrained_model == 'nuclei':
            model = models.CellposeModel(gpu=gpu,model_type='nuclei')
        elif pretrained_model == 'cyto':
            model = models.CellposeModel(gpu=gpu,model_type='cyto')
        if not pretrained_model:
            model = models.CellposeModel(gpu=gpu,pretrained_model=None)
    
    try:
        model.train(train_data,train_labels,train_images,test_data,test_labels,test_images,channels =channels,
                rescale=rescale,learning_rate=lr,save_path=image_path, batch_size=batch_size,
                n_epochs=nepochs,save_each=save_each,save_every=save_every,model_name=model_name)
    except KeyboardInterrupt:
            print('Training interrupted.')

    if return_model == True:
        return model 
    
def predict_single_image(image_path, model,channels=[0,0], diameter=None,
                         min_size=15, rescale=None, config=None, return_results=False,
                         mute=False, save_masks=True,tar_dir='',model_id='',cp_version=__cp_version__):
    '''
    Segment one or multiple images with a trained model.
    '''

    if not isinstance(image_path, list):
        image_path = [str(Path(image_path).as_posix())]
    else:
        image_path = [str(Path(x).as_posix()) for x in image_path]

    try:
        img = [io.imread(str(x)) for x in image_path]
        img_id = [Path(x).stem for x in image_path]

        if cp_version > 3:
            channels = None

        if config:
            try:
                eval_str = ''
                for key,val in config.items():
                    if not eval_str:
                        i_str=f'{key}={val}'
                    else:
                        i_str=f',{key}={val}'
                    eval_str+=i_str
                exec(f'masks, flows, styles = model.eval(img, diameter=diameter,rescale=rescale,min_size=min_size,channels=channels, {eval_str})')
            except AttributeError:
                print('Config file is not formatted correctly. Please check the documentation for more information.')
            except SyntaxError:
                print('Diameter,rescale,min_size,channels are not allowed to be overwritten.')
        else:
            masks, flows, styles = model.eval(img, diameter=diameter, rescale=rescale, min_size=min_size, channels=channels); 
        
        if save_masks == False and return_results == False:
            print('Saving and returning of results were switched of - therefore mask saving was turned on!')
            save_masks = True

        if save_masks == True:
            if tar_dir:
                os.makedirs(Path(tar_dir), exist_ok=True)
                parent_folder = Path(tar_dir)
            else:
                parent_folder = Path(image_path[0]).parent.joinpath('predictions')
                os.makedirs(parent_folder, exist_ok=True)
            
            for ind, id in enumerate(img_id):
                io.imsave(parent_folder.joinpath(f'{id}_{model_id}_pred.tif'),masks[ind])

        if mute== False:
            print('Sucessfully created predictions for one image(s).')
    except KeyboardInterrupt:
        print('Aborted.')
    
    if return_results == True:
        return masks, flows, styles
    else:
        return None

def predict_folder(image_path, model, image_format='jpg', filter_str='',
                   channels=[0,0], diameter=None, min_size=15, rescale=None,
                   config=None,tar_dir='', save_masks=True,return_results=True,
                   mute=False, model_id='',cp_version=__cp_version__):
    """
    This function takes in a directory containing images, and uses a pre-trained model to predict segmentation masks for the images.
    If `return_results` is `True` respective lists of 1D arrays for predicted *masks*, *flows* and *styles* 
    from `CellposeModel.eval()` are returned (see https://cellpose.readthedocs.io/en/latest/api.html#id5).

    Parameters:
    ------------
    image_path (str, Path) - Input directory 
    model (obj) - Trained model from 'models.CellposeModel' class. 
        Use either `models.CellposeModel(model_type='')` for built-in cellpose models or 
        `models.CellposeModel(pretrained_model='') for custom models.
        See https://cellpose.readthedocs.io/en/latest/models.html for more details
    image_format (str(optional, default 'jpg')) - Image format of the images in `image_path`
    filter_str (str(optional, default '')) - A string used to filter the images in `image_path`
    config (dict(optional, default None)) - dictionary of advanced parameters to be handed down to `CellposeModel.eval()` where keys are parameters and values are parameter values.
    tar_dir (str(optional, default '')) - The directory to save the predicted masks to.
    save_masks (bool(optional, default True)) - flag for saving predicted mask as `.tif` files in `tar_dir`
    mute (bool (optional, default=False)) - flag for muting console output
    model_id (str (optional, default = '')) - optional model name that will be written into output file names
    use_bfloat16 (bool (optional, default=True)) - If True, the model is trained using float16 precision, which limits the number of objects to <65,535 per mask. Set to True for faster training.

    Parameters that can be handed down explicitly to `CellposeModel.eval()`, 
    see https://cellpose.readthedocs.io/en/latest/api.html#id25 :

    channels (list (optional, default [0,0]))
    diameter (float (optional, default None))
    rescale (float (optional, default None))
    min_size (int (optional, default 15))      
    
    Returns
    ------------
    mask_l (list of 2D array lists (optional, default = [])) - labelled image, where 0=no masks; 1,2,…=mask labels
    flow_l (list of 2D array lists (optional, default = [])) - flows[k][0] = XY flow in HSV 0-255 flows[k][1] = XY flows at each pixel 
        flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics) 
        flows[k][3] = final pixel locations after Euler integration
    styles_l (list of 1D arrays of length 64 (optional, default = [])) - style vector summarizing each image, also used to estimate size of objects in image
    id_list (list of strings (optional, default = [])) - Name tags for input images
    img_l (list 2D array lists (optional, default = [])) - Input images

    """

    image_path = str(Path(image_path).as_posix()) #ensure that Path is a string for cellpose classes
    mask_l,flow_l,styles_l,id_list,img_l = [],[],[],[],[]

    try:
        img_l = natsorted(glob(f'{Path(image_path)}/*{filter_str}*.{image_format}'))
        id_list = [Path(file).stem for file in img_l]

        if mute== False:
            print('Predicting for ',image_path,'...')
            
        for count, file in enumerate(tqdm(img_l, desc=image_path, unit='image', colour='CYAN')):
            if any(x in id_list[count] for x in ['flow','flows','masks','mask','pred','composite']):
                continue
            mask, flow, style = predict_single_image(
                image_path=file,
                model=model,
                channels=channels,
                diameter=diameter,
                min_size=min_size,
                rescale=rescale,
                config=config,
                return_results=True,
                save_masks=save_masks,
                mute=mute,
                tar_dir=tar_dir,
                model_id=model_id)

            mask_l.append(mask)
            flow_l.append(flow)
            styles_l.append(style)

        if mute== False:
            print(f'Sucessfully created predictions for', {count},'image(s).')
    except KeyboardInterrupt:
        print('Aborted.')

    if return_results == True:
        return mask_l, flow_l, styles_l, id_list, img_l
    else: 
        mask_l, flow_l, styles_l, id_list, img_l = [],[],[],[],[]

        return mask_l, flow_l, styles_l, id_list, img_l

def predict_dataset(image_path, model,image_format='jpg', channels=[0,0],
                    diameter=None, min_size=15, rescale=None, config=None, tar_dir='',
                    return_results=False, save_masks=True, mute=False,use_bfloat16=True,
                    do_subfolders=False, model_id='',use_gpu=True,cp_version=__cp_version__):
    """
    Wrapper for helper.prediction.predict_folder() for a dataset that is organised in subfolders (e.g., in directories named `train`,`test`)

    Parameters:
    ------------
    do_subfolders (bool (optional, default=False)) - flag to look for files in subfolders
    return_results (bool(optional, default False)) - flag for returning predicted masks, flows and styles

    all others are the same as helper.prediction.predict_folder()

    Returns
    ------------
    mask_ll (list of 2D array lists (optional, default = [])) - labelled image, where 0=no masks; 1,2,…=mask labels

    flow_ll (list of 2D array lists (optional, default = [])) - flows[k][0] = XY flow in HSV 0-255 flows[k][1] = XY flows at each pixel 
        flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics) 
        flows[k][3] = final pixel locations after Euler integration

    styles_ll (list of 1D arrays of length 64 (optional, default = [])) - style vector summarizing each image, also used to estimate size of objects in image
    
    list_of_id_lists (list of strings (optional, default = [])) - Name tags for input images

    img_ll (list 2D array lists (optional, default = [])) - Input images

    """

    mask_ll,flow_ll,styles_ll,list_of_id_lists,=[],[],[],[]

    if type(model) != models.CellposeModel:
        try:
            model_path = str(Path(model).as_posix())
            if cp_version > 3:
                model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path, use_bfloat16=use_bfloat16)
            else:
                model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
            if not model_id:
                model_id = Path(model_path).stem
        except Exception as err:
            print(f"Could not load model. {err}: / Please check Path or switch to CPU in case of not enough GPU memory.")
            return
        
    working_directories = data_loader.assert_work_dirs(image_path,do_subfolders=do_subfolders)

    for working_directory in working_directories:
        check_l = natsorted(glob(f'{working_directory}/*.{image_format}'))

        if len(check_l)>0:
            working_directory = str(Path(working_directory).as_posix()) #ensure that working directory is a string for cellpose classes
            mask_l_i,flow_l_i,styles_l_i,id_list_i,_ = predict_folder(working_directory,model,image_format=image_format,channels=channels,diameter=diameter,
            min_size=min_size,rescale=rescale,config=config,tar_dir=tar_dir,save_masks=save_masks,mute=mute,model_id=model_id)
            if return_results==True:
                for idx,x in enumerate(mask_l_i):
                    mask_ll.append(x)
                    flow_ll.append(flow_l_i[idx])
                    styles_ll.append(styles_l_i[idx])
                    list_of_id_lists.append(id_list_i[idx])
        else:
            continue
    
    return mask_ll,flow_ll,styles_ll,list_of_id_lists

def batch_predict(model_dir, dir_paths, configuration=None, image_format='jpg',use_bfloat16=True,
                  use_GPU=True, channels=[0,0], diameter=None, min_size=15,
                  rescale=None, tar_dir='', return_results=False, save_masks=True,
                  mute=False, do_subfolders=False, cp_version=__cp_version__):
    """
    Wrapper for helper.prediction.predict_dataset() that can do predictions on the same dataset for multiple models from a directory (`model_dir`).

    Parameters:
    ------------
    dir_paths (list of str, Path) - list of images to segment 
    model_dir (str, Path) - model directory 
    use_GPU (bool (optional, default=True)) - GPU flag
    configuration (dict or list of dicts (optional, default = None))
        dictionary where `key` = paramter name and `val` = parameter value; can be varied for each cellpose model model in `model_dir`
        currently handed down are:
            channels (list (optional, default [0,0]))
            diameter (float (optional, default None))
            rescale (float (optional, default None))
            min_size (int (optional, default 15))

    all others are the same as helper.prediction.predict_dataset()

    Returns
    ------------
    all_results (dict (optional, default = {})) - dict containing output from helper.prediction.predict_dataset().
    
    """

    model_dir = str(Path(model_dir).as_posix())

    if cp_version > 3:
        if Path(model_dir).suffix == '' and 'cp_SAM' in str(model_dir):
            model_list = [model_dir]
            model_id_list = [Path(model_dir).stem]
        else:
            model_list,model_id_list = models_from_zoo(model_dir)
    else:                   
        if '.' in str(model_dir):
            model_list = [model_dir]
            model_id_list = [Path(model_dir).stem]
        else:
            model_list,model_id_list = models_from_zoo(model_dir)

    all_results= {}

    for m_idx in range(len(model_list)):
        if cp_version > 3:
            model = models.CellposeModel(gpu=use_GPU,pretrained_model=str(model_list[m_idx]),use_bfloat16=use_bfloat16)
        else:
            model = models.CellposeModel(gpu=use_GPU,pretrained_model=str(model_list[m_idx]))
        model_id = model_id_list[m_idx]
        print(model_id,'found...')

        if configuration:
            if len(configuration)>1:
                try:
                    config = configuration[m_idx]
                except AttributeError:
                    pass
            else:
                if type(configuration)==dict:
                    config = configuration
        else:
            config = None

        if type(dir_paths) != list:
            dir_paths = [dir_paths]
        dir_paths = [str(Path(dir_paths[idx]).as_posix()) for idx in range(len(dir_paths))]

        for d_idx in range(len(dir_paths)):
            all_mask_l,all_flow_l,all_styles_l,all_id_list = predict_dataset(dir_paths[d_idx],model,
            image_format=image_format,channels=channels,diameter=diameter,min_size=min_size,rescale=rescale,config=config,tar_dir=tar_dir,
            return_results=return_results,save_masks=save_masks,mute=mute,do_subfolders=do_subfolders,model_id=model_id,
            use_bfloat16=use_bfloat16,use_gpu=use_GPU,cp_version=cp_version)
            if return_results == True:
                dataset_res = {'masks':all_mask_l,'flows':all_flow_l,'styles':all_styles_l,'id':all_id_list}
                all_results[f'{model_id}_{d_idx}']=dataset_res

    return all_results

def models_from_zoo(model_dir, use_GPU=True,cp_version=__cp_version__,use_bfloat16=True):
    """
    Loads pre-trained cellpose model(s) from a folder.

    Parameters:
    ------------
    model_dir (str, Path) - model directory 
    use_GPU (bool (optional, default=True)) - GPU flag
    cp_version (int (optional, default=__cp_version__)) - Cellpose version
    use_bfloat16 (bool (optional, default=True)) - BFloat16 flag

    Returns
    ------------
    model_list (list) - list of cellpose model paths
    model_id_list (list) - list of cellpose model names

    """

    if cp_version > 3:
        model_list = natsorted(glob(f'{Path(model_dir)}/*cp_SAM*'))
    else:
        model_list = natsorted(glob(f'{Path(model_dir)}/*.*'))

    try:
        if cp_version > 3:
            model = models.CellposeModel(gpu=use_GPU,pretrained_model=model_list[0],use_bfloat16=use_bfloat16)  
        else:
            models.CellposeModel(gpu=use_GPU,pretrained_model=model_list[0])
    except:
        print('No cellpose model found in this directory.')

    model_id_list = [Path(model_list[idx]).stem for idx in range(len(model_list))]

    return model_list,model_id_list

def combine_preds(preds_small, preds_large, imgs,tar_dir='', model_id='',
                  filters=None, threshold=None, mute=True, do_composites=True,
                  remove_intersecting=False, stack_3D=False, file_name=''):
    """
    Combines predictions from two models (small and large) into one mask.

    """

    if tar_dir != '':
        os.makedirs(tar_dir,exist_ok=True)

    if stack_3D==False:    
        if type(imgs) == np.ndarray and imgs.ndim > 2:
                print('Numpy.ndarray passed - trying 3D combination!')
                stack_3D = True
    print(f'Combining predictions for {len(imgs)} images:')

    if stack_3D == True:
        combine_3D(preds_small,preds_large,tar_dir=tar_dir,model_id=model_id,filters=filters,threshold=threshold,mute=mute,
                  remove_intersecting=remove_intersecting,file_name=file_name)
    else:
        combine_2D(preds_small,preds_large,imgs,tar_dir=tar_dir,model_id=model_id,filters=filters,threshold=threshold,mute=mute,
                  do_composites=do_composites,remove_intersecting=remove_intersecting)
        
    return

def combine_3D(preds_small, preds_large, tar_dir='', model_id='', filters=None,
               threshold=None, mute=True, remove_intersecting=False, file_name=''):
    """
    Combines predictions from two models (small and large) into one mask in 3D.
    """

    if type(preds_small) != np.ndarray:
        print('No Numpy.ndarray passed - cannot do 3D!')
        return
    stack_list = []
    conflict_list = []

    if file_name != '':
        file_id = file_name
    else:
        file_id = 'Stack_3D'
        print('No file name provided - will use: Stack_3D!')
    #adapt label numbers to ensure no duplicates in 3D
    max_label = np.max(preds_large)
    preds_small = np.where(preds_small > 0, preds_small + max_label, preds_small)

    for masks1,masks2 in tqdm(zip(preds_small,preds_large),desc=str(file_name),unit='slice'):
        if threshold != None:
        #filter first with regular quality filters and then split along size_threshold
            m1,_ = grainsizing.filter_by_threshold_size(masks1,mute=mute,filters=filters,threshold=threshold,remove='small')
            m2,_ = grainsizing.filter_by_threshold_size(masks2,mute=mute,filters=filters,threshold=threshold,remove='large')
        elif filters != None:
        #filter first with regular quality filters 
            _, m1 = grainsizing.filter_grains(labels=masks1, filters=filters,mask=masks1,mute=mute)
            _, m2 = grainsizing.filter_grains(labels=masks1, filters=filters,mask=masks2,mute=mute)
        else: 
            m1 = masks1
            m2 = masks2

        if not any(x is None for x in [m1,m2]): 
            #combine masks
            combined = np.where(m2 == 0, m2 + m1, m2) #simple priority for large grains
            if remove_intersecting == True:
                m3 = np.where(m2 > 0, m1, m2) #create array for intersecting preds
                conflicting_labels = np.unique(m3) #get labels of conflicting/intersecting grains
                conflict_list= np.concatenate((conflict_list, conflicting_labels), axis=None)
                conflict_list = np.unique(conflict_list)
        else:
            combined = m1
        stack_list.append(combined)
    new_stack = np.stack(stack_list)

    if remove_intersecting == True:
        for key in conflict_list:
            new_stack[new_stack == key]=0  #remove entire grain for intersecting preds in 3D

    if tar_dir == '':
            data_path= f'{Path.cwd().as_posix()}/predictions/'
            os.makedirs(data_path,exist_ok=True)
    else:
        data_path = tar_dir

    filename_i = f'{data_path}/{file_id}_{model_id}_combined_pred.tif'
    imwrite(filename_i, new_stack)

    return

def combine_2D(preds_small, preds_large, imgs,tar_dir='', model_id='',
               filters=None, threshold=150, mute=True, do_composites=True,
               remove_intersecting=False):
    """
    Combines predictions from two models (small and large diameters) into one mask in 2D.
    """

    for p_1,p_2,img in tqdm(zip(preds_small,preds_large,imgs),unit='image'):
        #load preds for small grains
        masks1 = io.imread(p_1)
        #load preds for large grains
        masks2 = io.imread(p_2)
        file_id = Path(img).stem
        #filter first with regular quality filters and then split along size_threshold
        m1,_ = grainsizing.filter_by_threshold_size(masks1,mute=mute,filters=filters,threshold=threshold,remove='large')
        m2,props2 = grainsizing.filter_by_threshold_size(masks2,mute=mute,filters=filters,threshold=threshold,remove='small')
        
        if not any(x is None for x in [m1,m2,props2]):
            #adapt label numbers to ensure no duplicates
            m1 = np.where(m1 > 0, m1 + np.max(props2['label']), m1) # use not length but max value!
            if remove_intersecting == True:
                m3 = np.where(m2 > 0, m1, m2) #create array for intersecting preds
                conflicting_labels = np.unique(m3) #get labels of conflicting/intersecting grains
                for key in conflicting_labels:
                    m1[m1 == key]=0  #remove entire grain for intersecting preds from small preds
            #combine masks
            combined = np.where(m2 == 0, m2 + m1, m2) #simple priority for large grains
        else:
            print(f'Could not combine preds for {file_id} due to an empty prediction! Using grains from one inferrence only.')
            combined = m1
        if tar_dir == '':
                data_path=f'{Path(img).parent}/predictions/'
                os.makedirs(data_path,exist_ok=True)
        else:
            data_path = tar_dir
        filename_i = f'{data_path}/{file_id}_{model_id}_combined_pred.tif'
        cv2.imwrite(filename_i, combined)
        if do_composites == True:
            plotting.do_composite(img,filename_i,data_path,file_id,model_id=f'{model_id}_combined', tar_dir=tar_dir)
        if mute == False:
            print(file_id)

    return 

def eval_image(y_true, y_pred, thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]):
    """
    Evaluates a single image. Uses cellpose.metrics (https://cellpose.readthedocs.io/en/latest/api.html#module-cellpose.metrics).
    
    Parameters:
    ------------
    y_true (array) - ground truth mask
    y_pred (array) - predicted mask
    thresholds (list (optional, default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])) - Thresholds to evaluate at
    
    Returns
    ------------
    ap (float) - average precision
    tp (float) - true positives
    fp (float) - false positives
    fn (float) - false negatives
    iout (float) - intersection over union
    preds (float) - predicted mask
    
    """

    ap, tp, fp, fn = metrics.average_precision(label(y_true),label(y_pred),threshold=thresholds)
    iout, preds = metrics.mask_ious(label(y_true),label(y_pred))

    return ap, tp, fp, fn, iout, preds

def eval_set(imgs, lbls, preds, data_id='', tar_dir='',
             thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
             filters={'edge':[False,.05],'px_cutoff':[False,10]},
             filter_props=['label','area','centroid','major_axis_length','minor_axis_length'],
             save_results=True, return_results=True, return_test_idx=False, mute=True):
    """
    Evaluates a set of images with eval_image. Saves results to a pkl file.

    Parameters:
    ------------
    imgs (list) - List of images paths
    lbls (list) - List of labels paths
    preds (list) - List of predictions paths
    data_id (str (optional, default='')) - ID for the dataset
    tar_dir (str (optional, default='')) - Directory to save results to
    thresholds (list (optional, default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])) - Thresholds to evaluate at
    filters (dict (optional, default={'edge':[False,.05],'px_cutoff':[False,10]})) - Dictionary of filters to apply to labels and predictions
    filter_props (list (optional, default=['label','area','centroid','major_axis_length','minor_axis_length'])) - Properties to filter on
    save_results (bool (optional, default=True)) - Flag whether to save results to a pkl file
    return_results (bool (optional, default=True)) - Flag whether to return results

    Returns
    ------------
    eval_results (dict) - Dictionary of evaluation results

    """
    eval_results={}

    for idx,im in enumerate(imgs):
        img_id = Path(im).stem
        img = io.imread(str(im))
        y_true = io.imread(str(lbls[idx]))
        y_pred = io.imread(str(preds[idx]))

        if filters:
            if np.unique(label(y_true)).any() > 0: #check if labels are not empty
                _, y_true = grainsizing.filter_grains(labels=y_true,properties=filter_props,filters=filters,mask=y_true)
            else:
                if mute == False:
                    print('!Empty labels for image: ',lbls[idx],' - check if predictions were passed as labels!')
                continue
                
            if np.unique(label(y_pred)).any() > 0: #check if prediction is not empty
                _, y_pred = grainsizing.filter_grains(labels=y_pred,properties=filter_props,filters=filters,mask=y_pred)
                ap,tp,fp,fn,iout,_ =  eval_image(y_true,y_pred, thresholds=thresholds)
            else:
                if mute == False:
                    print('! Empty prediction for image: ',preds[idx],' !')
                iout = []
                tp = np.zeros(len(thresholds))
                fp = np.zeros(len(thresholds))
                fn = np.zeros(len(thresholds))
                ap = np.zeros(len(thresholds))

        eval_results[idx] = {'id':img_id,'img':img, 'ap':ap, 'iout':iout,'tp':tp,'fp':fp,'fn':fn}

    if save_results==True:
        if tar_dir != '':
            os.makedirs(Path(tar_dir), exist_ok=True)
            export = f'{tar_dir}/{data_id}_eval_res.pkl'
        else:
            try:
                parent_dir = str(Path(imgs[0]).parent.parent.as_posix())
                export = f'{parent_dir}/{data_id}_eval_res.pkl'
            except:
                export = f'{data_id}_eval_res.pkl'
        with open(str(export), 'wb') as f:
            pickle.dump(eval_results, f)

    if return_results == True:
        return eval_results
    
def eval_wrapper(pred_list, imgs, filterstrings, taglist, filters=None,
                 save_results=True, m_string='_mask',dataset='', out_path=''):
    """ 
    Wrapper for eval_set to evaluate multiple predictions on the same dataset
    """
    res_list, tt_list,preds_fil_sort_list = [],[],[]

    for i in range(len(pred_list)):
        preds_fil_sort = map_preds_to_imgs(pred_list[i],imgs,p_string=filterstrings[i],m_string=m_string)
        test_idxs = find_test_idxs(imgs)
        i_res = eval_set(imgs,imgs,preds_fil_sort,
                                            data_id = f'{out_path}{taglist[i]}_on_{dataset}',filters=filters, save_results=save_results)
        res_list.append(i_res)
        tt_list.append(test_idxs)
        preds_fil_sort_list.append(preds_fil_sort)

    return res_list, tt_list, preds_fil_sort_list

def map_preds_to_imgs(preds, imgs, p_string='', m_string=''):
    """ 
    Match predictions to images/labels based on the file name.

    Parameters:
    ------------
    preds (list) - List of predictions paths
    imgs (list) - List of images paths
    p_string (str (optional, default='')) - String to split the prediction file name
    m_string (str (optional, default='')) - String to split the image file name

    Returns
    ------------
    new_preds (list) - List of matched predictions paths
    """

    new_preds = []
    for kk in range(len(imgs)):
        if m_string:
            img_id = Path(imgs[kk]).stem.split(m_string)[0]
        else:
            img_id = Path(imgs[kk]).stem
        for k in range(len(preds)):
            if p_string:
                img_id2 = Path(preds[k]).stem.split(p_string)[0]
            else:
                img_id2 = Path(preds[k]).stem
            if img_id == img_id2:
                new_preds.append(preds[k])

    if not new_preds:
        print(p_string,' - Could not match prediction to images!')

    return new_preds

def find_test_idxs(lbls):
    """
    Find the indices of the test images in the list of labels.

    Parameters:
    ------------
    lbls (list) - List of labels paths
    
    Return
    ------------
    test_idxs (list) - List of indices of the test images
    """ 

    test_idxs = []

    for idx, x in enumerate(lbls):
        if 'test' in x:
            test_idxs.append(idx)

    return test_idxs

def map_res_to_imgs(eval_results, imgs):
    """
    Match results to images based on the file name.
    """

    new_res = {}

    for kk in range(len(imgs)):
        img_id = Path(imgs[kk]).stem
        for k in range(len(eval_results)):
            if img_id == eval_results[k]['id']:
                new_res[kk] = eval_results[k]

    return new_res

def get_stats_for_res(preds, eval_results, test_idxs=None,return_prec_recall=False):
    """
    Get average precision stats.

    Parameters:
    ------------
    preds (list) - List of predictions paths
    eval_results (dict) - Dictionary of evaluation results
    test_idxs (list (optional, default=None)) - Indices of images in test split

    Returns
    ------------
    res_stats (list) - List of average precision stats
    """

    tpreds, taps50, tamaps, t_tp, t_fp, t_fn = [],[],[],[],[],[]
    ttpreds, ttaps50, ttamaps,tt_tp, tt_fp, tt_fn = [],[],[],[],[],[]


    for i in range(len(preds)):
        a = regionprops_table(label((io.imread(str(preds[i])))))
        napred = len(a['label'])
        aap50 = eval_results[i]['ap'][0]
        amap = eval_results[i]['ap'][0:9].mean()
        tpap50 = eval_results[i]['tp'][0]
        fpap50 = eval_results[i]['fp'][0]
        fnap50 = eval_results[i]['fn'][0]
        
        if test_idxs:

            if i < len(test_idxs):
                tpreds.append(napred)
                taps50.append(aap50)
                tamaps.append(amap)
                t_tp.append(tpap50)
                t_fp.append(fpap50)
                t_fn.append(fnap50)
            else:
                ttpreds.append(napred)
                ttaps50.append(aap50)
                ttamaps.append(amap)
                tt_tp.append(tpap50)
                tt_fp.append(fpap50)
                tt_fn.append(fnap50)
        else:
            ttpreds.append(napred)
            ttaps50.append(aap50)
            ttamaps.append(amap)
            tt_tp.append(tpap50)
            tt_fp.append(fpap50)
            tt_fn.append(fnap50)

    res_stats = [np.sum(tpreds),np.mean(taps50),np.std(taps50),np.mean(tamaps),np.std(tamaps),
                  np.sum(ttpreds),np.mean(ttaps50),np.std(ttaps50),np.mean(ttamaps),np.std(ttamaps)]

    if return_prec_recall == True:
        res_stats2 =[np.sum(t_tp)/(np.sum(t_tp)+np.sum(t_fp)), np.sum(t_tp)/(np.sum(t_tp)+np.sum(t_fn))]
        return (res_stats, res_stats2)
    else:
        return res_stats


def get_stats_for_run(pred_list, res_list, titles,
                      p_string_list, labels, test_idxs_list=None):
    """
    Use to define.
    """

    cols = ['model','n_pred_test','mAP50_test','std','mAP50_90_test','std','n_pred_train','mAP50_train','std','mAP50_90_train','std']
    res_stats = pd.DataFrame(columns=cols)

    for j in range(len(pred_list)):
        sorted = map_preds_to_imgs(pred_list[j],labels,p_string=p_string_list[j],m_string='_mask')
        if test_idxs_list:
            entry = get_stats_for_res(sorted,res_list[j],test_idxs=test_idxs_list[j])
        else:
            entry = get_stats_for_res(sorted,res_list[j])
        res_stats.loc[j] = [titles[j]]+entry

    return res_stats

def get_style_vectors(do_inference=True, tar_dir='', model='default', use_gpu=True,
                      im_paths=None, mute = True,res_file=None,cp_version=__cp_version__,use_bfloat16=True):
    """
    Use to define.
    """

    if model == 'default':
        homepath = Path.home().joinpath('imagegrains')
        if cp_version > 3:
            model = f'{homepath}/models/IG2_full_set_cp_SAM'
        else:   
            model = f'{homepath}/models/IG2_full_set.200525'

    if not model:
        try:
            if cp_version > 3:
                model = f'{homepath}/models/IG2_full_set_cp_SAM'
            else:   
                model = f'{homepath}/models/IG2_full_set.200525'
        except:
            pass

    if tar_dir:
        os.makedirs(tar_dir, exist_ok=True)

    if do_inference == True:
        if im_paths:
            print(f'Running inference for styles with {Path(model).name}...')
            if cp_version > 3:
                model = models.CellposeModel(gpu=use_gpu,pretrained_model=model,use_bfloat16=use_bfloat16)
            else:
                model = models.CellposeModel(gpu=use_gpu,pretrained_model=model)
            test_styles,train_styles,testnames,trainnames,testpaths,trainpaths = [],[],[],[],[],[]

            for path in im_paths:
                _,_,styles,ids,imgs=predict_folder(path,model,min_size=-1, mute=True,return_results=True)
                for styli,idi,imgi in zip(styles,ids,imgs):
                    imgi = Path(imgi).as_posix()
                    if '/test' in imgi:
                        test_styles.append(styli)
                        testnames.append(idi)
                        testpaths.append(imgi)
                    else:
                        train_styles.append(styli)
                        trainnames.append(idi)
                        trainpaths.append(imgi)
            res_dict = {'train':train_styles, 'train_ids':trainnames,'train_paths':trainpaths,
                        'test':test_styles,'test_ids':testnames,'test_paths':testpaths}
            with open(f'{tar_dir}/res_tSNE.pkl', 'wb') as handle:
                pickle.dump(res_dict, handle)
        else: 
            print('No path to images provided!')
            return [],[],[],[],[],[]
    else:

        if not res_file:
            try: 
                homepath = homepath = Path.home().joinpath('imagegrains')
                res_file = f'{homepath}/demo_data/res_tSNE.pkl'
            except:
                print('Could not find file with style vectors!')
        res_file=Path(res_file).as_posix()
        pkl_file = open(f'{res_file}', 'rb')
        tres = pickle.load(pkl_file)
        
        train_styles = tres['train']
        trainnames = tres['train_ids']
        
        try:
            test_styles = tres['test']
            testnames = tres['test_ids']
        except:
            test_styles,testnames= [],[]
        
        try:
            testpaths = tres['test_paths']
            trainpaths = tres['train_paths']
        except:
            trainpaths,testpaths = [],[]
        pkl_file.close()

    return train_styles, trainnames, trainpaths, test_styles, testnames,testpaths

def keep_tif_crs(imgs, preds, mute=True):
    """
    Keep the coordinate reference system (CRS) of the original image when saving the predictions.
    This is done by copying the tfw file and the georeference from the original image to the prediction.
    
    Parameters:
    ------------
    imgs (list) - List of images paths
    preds (list) - List of predictions paths
    mute (bool (optional, default=True)) - Flag for muting console output
    
    """

    try: 
        from osgeo import gdal
        gdal.UseExceptions()

        try:
            for im,pr in zip(imgs,preds):
                #copy tfw file if existing
                src_tfw = f'{Path(im).parent.as_posix()}/{Path(im).stem}.tfw'
                tar_tfw = f'{Path(pr).parent.as_posix()}/{Path(pr).stem}.tfw'
                shutil.copy(src_tfw,tar_tfw)
                #copy georeference if existing
                dataset = gdal.Open(im)
                projection   = dataset.GetProjection()
                geotransform = dataset.GetGeoTransform()
                #update metadata
                dataset2 = gdal.Open(pr, gdal.GA_Update)    
                dataset2.SetProjection( projection )
                dataset2.SetGeoTransform( geotransform )             
                #close raster files
                dataset = None
                dataset2 = None
        except:
            if mute == False:
                print('>> Georeference of tif/tiff files incomplete. Predictions might not be correctly referenced.')
            pass

    except ModuleNotFoundError:
        if mute == False:
            print('>> GDAL not installed. Please install GDAL to keep CRS info for GeoTIFF files.')
        pass