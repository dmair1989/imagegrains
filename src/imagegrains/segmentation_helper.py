import os,pickle
from pathlib import Path
from cellpose import io
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from skimage.measure import label, regionprops_table
import numpy as np
import pandas as pd

from cellpose import metrics
from cellpose import models
from cellpose.io import logger_setup

from imagegrains import import grainsizing
from imagegrains import import data_loader


def check_labels(labels,TAR_DIR='',lbl_str='_mask',mask_format='tif'):
    """
    This function checks if the labels are in the correct format. If not, it renames the labels to the correct format.
    The labels are renamed to the format: <image_ID><lbl_str>.<mask_format>
    
    Parameters:
    ------------
    labels (list) - List of paths to the labels
    TAR_DIR (str(optional, default '')) - Target directory to save the renamed labels. If not specified, the labels are saved in the same directory as the original labels.
    lbl_str (str(optional, default '_mask')) - String to be added to the image ID to form the label ID
    mask_format (str(optional, default 'tif')) - Image format of the labels

    Returns:
    ------------
    track_l (list) - List of image IDs for which the labels were renamed
    
    """
    if TAR_DIR:
        os.makedirs(TAR_DIR, exist_ok=True)
    track_l = []
    for label in labels:
        if lbl_str in label:
            continue
        else:
            img= io.imread(label)
            ID = Path(label).stem
            #ID = label.split('\\')[len(label.split('\\'))-1].split('.')[0]
            print(ID)
            #plt.imshow(img)
            io.imsave(TAR_DIR+'/'+ID+lbl_str+'.'+mask_format,img)
            track_l.append(ID)
    if len(track_l) == 0:
        print('No files renamed.')
    return track_l

def check_im_label_pairs(img_list,lbl_list):
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
        ID = Path(image).stem
        #ID = image.split('\\')[len(image.split('\\'))-1].split('.')[0]
        if any(ID in x for x in lbl_list):
            continue
        else:
            error_list.append(image)
    if len(error_list)==0:
        print('All images have labels.')
    return error_list

def custom_train(PATH, pretrained_model = None,datstring = None,
                lr = 0.2, nepochs = 1000,chan1 = 0, chan2= 0, gpu = True, batch_size = 8,
                mask_filter = '_mask', rescale = False, save_each = False, return_model = False,
                save_every = 100,model_name = None, label_check = True):
    
    """
    This function trains a model on the images and labels in the specified directory. The images and labels should be in the same directory. The labels should be in the format: <image_ID>_mask.<mask_format>
    
    Parameters:
    ------------
    PATH (str) - Path to the directory containing the images and labels
    pretrained_model (str(optional, default None)) - Path to the pretrained model. If not specified, the model is trained from scratch.
    datstring (str(optional, default None)) - String to be added to the model name. 
    return_model (bool(optional, default False)) - If True, the model is returned
    model_name (str(optional, default None)) - Name of the model.
    label_check (bool(optional, default True)) - If True, the labels are checked for the correct format and the images and labels are checked for correct pairing. If the labels are not in the correct format, they are renamed to the correct format. If the images and labels are not paired correctly, a list of images for which the labels are missing is returned.

    more parameters:
    https://cellpose.readthedocs.io/en/latest/api.html#cellpose.models.CellposeModel.train 

    Returns:
    ------------
    model (CellposeModel, optional) - Trained model
    
    """

    logger, log_file = logger_setup()
    train_images,train_masks,test_images,test_masks = data_loader.find_data(PATH,mask_str=mask_filter)
    if label_check == True:
        check_labels(train_masks);
        check_labels(test_masks);
        check_im_label_pairs(train_images,train_masks);
        check_im_label_pairs(test_images,test_masks);

    train_data,train_labels,test_data,test_labels = [],[],[],[]
    for x1,y1 in zip(train_images,train_masks):
        train_data.append(io.imread(x1))
        train_labels.append(io.imread(y1))

    for x2,y2 in zip(test_images,test_masks):
        test_data.append(io.imread(x2))
        test_labels.append(io.imread(y2))
    if not model_name:
        model_name = model_name
    else: 
        if not datstring:
            datstring = ''
        model_name = model_name + '.' + datstring
    if not pretrained_model:
        model = models.CellposeModel(gpu=gpu,pretrained_model=None)
    elif pretrained_model == 'nuclei':
        model = models.CellposeModel(gpu=gpu,model_type='nuclei')
    elif pretrained_model == 'cyto':
        model = models.CellposeModel(gpu=gpu,model_type='cyto')
    else:
        model = models.CellposeModel(gpu=gpu,pretrained_model=pretrained_model)
    try:
        model.train(train_data,train_labels,train_images,test_data,test_labels,test_images,channels =[chan1,chan2],
                rescale=rescale,learning_rate=lr,save_path=PATH, batch_size=batch_size,
                n_epochs=nepochs,save_each=save_each,save_every=save_every,model_name=model_name)
    except KeyboardInterrupt:
            print('Training interrupted.')
    if return_model == True:
        return model 

def predict_folder(INP_DIR,model,image_format='jpg',filter_str='',channels=[0,0],diameter=None,min_size=15,rescale=None,config=None,TAR_DIR='',
return_results=False,save_masks=True,mute=False,mID=''):
    """
    This function takes in a directory containing images, and uses a pre-trained model to predict segmentation masks for the images.
    If `return_results` is `True` respective lists of 1D arrays for predicted *masks*, *flows* and *styles* 
    from `CellposeModel.eval()` are returned (see https://cellpose.readthedocs.io/en/latest/api.html#id5).

    Parameters:
    ------------
    INP_DIR (str) - Input directory 
    model (obj) - Trained model from 'models.CellposeModel' class. 
        Use either `models.CellposeModel(model_type='')` for built-in cellpose models or 
        `models.CellposeModel(pretrained_model='') for custom models.
        See https://cellpose.readthedocs.io/en/latest/models.html for more details
    image_format (str(optional, default 'jpg')) - Image format of the images in `INP_DIR`
    filter_str (str(optional, default '')) - A string used to filter the images in `INP_DIR`
    return_results (bool(optional, default False)) - flag for returning predicted masks, flows and styles
    config (dict(optional, default None)) - dictionary of advanced parameters to be handed down to `CellposeModel.eval()` where keys are parameters and values are parameter values.
    TAR_DIR (str(optional, default '')) - The directory to save the predicted masks to.
    save_masks (bool(optional, default True)) - flag for saving predicted mask as `.tif` files in `TAR_DIR`
    mute (bool (optional, default=False)) - flag for muting console output
    mID (str (optional, default = '')) - optional model name that will be written into output file names

    Parameters that can be handed down explicitly to `CellposeModel.eval()`, 
    see https://cellpose.readthedocs.io/en/latest/api.html#id5 :

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
    ID_l (list of strings (optional, default = [])) - Name tags for input images
    img_l (list 2D array lists (optional, default = [])) - Input images

    """
    mask_l,flow_l,styles_l,ID_l,img_l = [],[],[],[],[]
    try:
        Z = natsorted(glob(INP_DIR+'/*'+filter_str+'*.'+image_format))
        if mute== False:
            print('Predicting for ',INP_DIR,'...')
        count=0
        #for _,im in tqdm(enumerate(Z), desc=INP_DIR,unit='image',colour='CYAN'):               
        for idx in tqdm(range(len(Z)), desc=INP_DIR,unit='image',colour='CYAN'):
            img= io.imread(Z[idx])
            ID = Path(Z[idx]).stem
            #ID = Z[im_idx].split('\\')[len(Z[im_idx].split('\\'))-1].split('.')[0]
            if any(x in ID for x in ['flow','flows','masks','mask','pred']):
                continue
            else:
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
                    masks, flows, styles = model.eval(img, diameter=diameter,rescale=rescale,min_size=min_size,channels=channels) 
                if save_masks == False and return_results == False:
                    print('Saving and returning of results were switched of - therefore mask saving was turned on!')
                    save_masks = True
                if save_masks == True:
                    if TAR_DIR:
                        os.makedirs(TAR_DIR, exist_ok=True)
                        io.imsave(TAR_DIR+'/'+ID+'_'+mID+'_pred.tif',masks)
                    else:
                        io.imsave(INP_DIR+'/'+ID+'_'+mID+'_pred.tif',masks)
                if return_results == True:
                    mask_l.append(masks)
                    flow_l.append(flows)
                    styles_l.append(styles)
                    ID_l.append(ID)
                    img_l = [Z[x] for x in range(len(Z))]
                count+=1
        if mute== False:
            print('Sucessfully created predictions for',count,'image(s).')
    except KeyboardInterrupt:
        print('Aborted.')
    return mask_l,flow_l,styles_l,ID_l,img_l

def predict_dataset(INP_DIR,model,image_format='jpg',channels=[0,0],diameter=None,min_size=15,rescale=None,config=None,TAR_DIR='',
return_results=False,save_masks=True,mute=False,do_subfolders=False,mID=''):
    """
    Wrapper for helper.prediction.predict_folder() for a dataset that is organised in subfolders (e.g., in directories named `train`,`test`)

    Parameters:
    ------------
    do_subfolders (bool (optional, default=False)) - flag to look for files in subfolders

    all others are the same as helper.prediction.predict_folder()

    Returns
    ------------
    mask_ll (list of 2D array lists (optional, default = [])) - labelled image, where 0=no masks; 1,2,…=mask labels

    flow_ll (list of 2D array lists (optional, default = [])) - flows[k][0] = XY flow in HSV 0-255 flows[k][1] = XY flows at each pixel 
        flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics) 
        flows[k][3] = final pixel locations after Euler integration

    styles_ll (list of 1D arrays of length 64 (optional, default = [])) - style vector summarizing each image, also used to estimate size of objects in image
    
    ID_ll (list of strings (optional, default = [])) - Name tags for input images

    img_ll (list 2D array lists (optional, default = [])) - Input images

    """
    mask_ll,flow_ll,styles_ll,ID_ll,=[],[],[],[]
    try:
        dirs = next(os.walk(INP_DIR))[1]
    except StopIteration:
        dirs=[]
        W_DIR = INP_DIR+'/'
        pass
    for dir in dirs:
        if dir=='train':
            W_DIR = INP_DIR+'/'+str(dir)+'/'
        elif dir=='test':
            W_DIR = INP_DIR+'/'+str(dir)+'/'
        elif do_subfolders == True:
            W_DIR = INP_DIR+'/'+str(dir)+'/'
        else:
            W_DIR = INP_DIR
        check_l = natsorted(glob(W_DIR+'/*.'+image_format))
        if len(check_l)>0:
            mask_l_i,flow_l_i,styles_l_i,ID_l_i,_ = predict_folder(W_DIR,model,image_format=image_format,channels=channels,diameter=diameter,
            min_size=min_size,rescale=rescale,config=config,TAR_DIR=TAR_DIR,return_results=return_results,save_masks=save_masks,mute=mute,mID=mID)
            if return_results==True:
                for idx,x in enumerate(mask_l_i):
                    mask_ll.append(x)
                    flow_ll.append(flow_l_i[idx])
                    styles_ll.append(styles_l_i[idx])
                    ID_ll.append(ID_l_i[idx])
        else:
            continue
    return mask_ll,flow_ll,styles_ll,ID_ll

def models_from_zoo(MOD_DIR,use_GPU=True):
    """
    Loads pre-trained cellpose model(s) from a folder.

    Parameters:
    ------------
    MOD_DIR (str) - model directory 
    use_GPU (bool (optional, default=True)) - GPU flag

    Returns
    ------------
    model_list (list) - list of cellpose model paths
    M_ID (list) - list of cellpose model names

    """
    model_list = natsorted(glob(MOD_DIR+'/*.*'))
    try:
        models.CellposeModel(gpu=use_GPU,pretrained_model=model_list[0])
    except:
        print('No cellpose model found in this directory.')
    M_ID = [Path(model_list[idx]).stem for idx in range(len(model_list))]
    #M_ID = [model_list[i].split('\\')[len(model_list[i].split('\\'))-1].split('.')[0] for i in range(len(model_list))]
    return model_list,M_ID

def batch_predict(MOD_DIR,DIR_PATHS,configuration=None,image_format='jpg',use_GPU=True,channels=[0,0],diameter=None,min_size=15,
rescale=None,TAR_DIR='',return_results=False,save_masks=True,mute=False,do_subfolders=False):
    """
    Wrapper for helper.prediction.predict_dataset() that can do predictions on the same dataset for multiple models from a directory (`MOD_DIR`).

    Parameters:
    ------------
    MOD_DIR (str) - model directory 
    use_GPU (bool (optional, default=True)) - GPU flag
    configuration (dict or list of dicts (optional, default = None))
        dictionary where `key` = paramter name and `val` = parameter value; can be varied for each cellpose model model in `MOD_DIR`
        currently handed down are:
            channels (list (optional, default [0,0]))
            diameter (float (optional, default None))
            rescale (float (optional, default None))
            min_size (int (optional, default 15))

    all others are thesame as helper.prediction.predict_dataset()

    Returns
    ------------
    all_results (dict (optional, default = {})) - dict containing output from helper.prediction.predict_dataset().
    
    """
    if '.' in MOD_DIR:
        model_list = [MOD_DIR]
        M_ID = [Path(MOD_DIR).stem]
    else:
        model_list,M_ID = models_from_zoo(MOD_DIR)
    all_results= {}
    for m_idx in range(len(model_list)):
        model = models.CellposeModel(gpu=use_GPU,pretrained_model=model_list[m_idx])
        mID = M_ID[m_idx]
        print(mID,'found...')
        if configuration:
            if len(configuration)>1:
                try:
                    config = configuration[m_idx]
                except AttributeError:
                    pass
            else:
                try:
                    for key,val in configuration.items():
                        exec(key + '=val')
                    if mute == False:
                        print('... with custom configuration.')
                except AttributeError:
                    pass
        else:
            config = None
        if type(DIR_PATHS) != list:
            DIR_PATHS = [DIR_PATHS] 
        for d_idx in range(len(DIR_PATHS)):
            all_mask_l,all_flow_l,all_styles_l,all_ID_l = predict_dataset(DIR_PATHS[d_idx],model,
            image_format=image_format,channels=channels,diameter=diameter,min_size=min_size,rescale=rescale,config=config,TAR_DIR=TAR_DIR,
            return_results=return_results,save_masks=save_masks,mute=mute,do_subfolders=do_subfolders,mID=mID)
            if return_results == True:
                dataset_res = {'masks':all_mask_l,'flows':all_flow_l,'styles':all_styles_l,'id':all_ID_l}
                all_results[str(mID)+'_'+str(d_idx)]=dataset_res
    return all_results


def eval_image(y_true,y_pred,thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]):
    """
    Evaluates a single image. Uses cellpose.metrics (https://cellpose.readthedocs.io/en/latest/api.html#module-cellpose.metrics).
    
    Parameters:
    ------------
    y_true (array) - ground truth mask
    y_pred (array) - predicted mask
    thresholds (list (optional, default=[0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])) - Thresholds to evaluate at
    
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
    #j_score = jaccard_score(y_true, y_pred,average="macro")
    #f1 = f1_score(y_true,y_pred,average="macro")
    return ap, tp, fp, fn, iout, preds

def eval_set(imgs,lbls,preds,dataID='',TAR_DIR='',thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
    filters={'edge':[False,.05],'px_cutoff':[False,10]},filter_props=['label','area','centroid','major_axis_length','minor_axis_length'],
    save_results=True,return_results=True,return_test_idx=False):
    """
    Evaluates a set of images with eval_image. Saves results to a pkl file.

    Parameters:
    ------------
    imgs (list) - List of images
    lbls (list) - List of labels
    preds (list) - List of predictions
    dataID (str (optional, default='')) - ID for the dataset
    TAR_DIR (str (optional, default='')) - Directory to save results to
    thresholds (list (optional, default=[0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])) - Thresholds to evaluate at
    filters (dict (optional, default={'edge':[False,.05],'px_cutoff':[False,10]})) - Dictionary of filters to apply to labels and predictions
    filter_props (list (optional, default=['label','area','centroid','major_axis_length','minor_axis_length'])) - pPoperties to filter on
    save_results (bool (optional, default=True)) - Flag whether to save results to a pkl file
    return_results (bool (optional, default=True)) - Flag whether to return results

    Returns
    ------------
    eval_results (dict) - Dictionary of evaluation results

    """
    eval_results={}
    for idx,im in enumerate(imgs):
        ID = Path(im).stem
        img = io.imread(im)
        y_true = io.imread(lbls[idx])
        y_pred = io.imread(preds[idx])

        if filters:
            _, y_true = grainsizing.filter_grains(labels=y_true,properties=filter_props,filters=filters,mask=y_true)
            if np.unique(label(y_pred)).any() > 0: #check if prediction is empty
                _, y_pred = grainsizing.filter_grains(labels=y_pred,properties=filter_props,filters=filters,mask=y_pred)
                ap,_,_,_,iout,_ =  eval_image(y_true,y_pred, thresholds=thresholds)
            else:
                print('! Empty prediction for image: ',preds[idx],' !')
                iout = []
                ap = np.zeros(len(thresholds))

        eval_results[idx] = {'id':ID,'img':img, 'ap':ap, 'iout':iout,}
    if save_results==True:
        if TAR_DIR:
            os.makedirs(TAR_DIR, exist_ok=True)
            export = TAR_DIR+'/'+dataID+'_eval_res.pkl'
        else:
            export = dataID+'_eval_res.pkl'
        with open(str(export), 'wb') as f:
            pickle.dump(eval_results, f)
    if return_results == True:
        return eval_results
    
def eval_wrapper(pred_list,imgs,filterstrings,taglist,filters=None,save_results=True,m_string='_mask',dataset='',out_path=''):
    """ 
    Wrapper for eval_set to evaluate multiple predictions on the same dataset
    """
    res_list, tt_list = [],[]
    for i in range(len(pred_list)):
        preds_fil_sort = map_preds_to_imgs(pred_list[i],imgs,p_string=filterstrings[i],m_string=m_string)
        test_idxs = find_test_idxs(imgs)
        i_res = eval_set(imgs,imgs,preds_fil_sort,
                                            dataID=str(out_path+taglist[i])+'_on'+str(dataset),filters=filters, save_results=save_results)
        res_list.append(i_res)
        tt_list.append(test_idxs)
    return res_list, tt_list, preds_fil_sort
    
def map_preds_to_imgs(preds,imgs,p_string='',m_string=''):
    """ 
    Match predictions to images/labels based on the file name.
    """
    new_preds = []
    for kk in range(len(imgs)):
        if m_string:
            ID = Path(imgs[kk]).stem.split(m_string)[0]
        else:
            ID = Path(imgs[kk]).stem
        for k in range(len(preds)):
            if p_string:
                ID2 = Path(preds[k]).stem.split(p_string)[0]
            else:
                ID2 = Path(preds[k]).stem
            if ID == ID2:
                new_preds.append(preds[k])
    if not new_preds:
        print(p_string,' - Could not match prediction to images!')
    return new_preds

def find_test_idxs(lbls):
    test_idxs = []
    for idx, x in enumerate(lbls):
        if 'test' in x:
            test_idxs.append(idx)
    return test_idxs

def map_res_to_imgs(res_dict,imgs):
    """
    Match results to images based on the file name.
    """
    new_res = {}
    for kk in range(len(imgs)):
        ID = Path(imgs[kk]).stem
        for k in range(len(res_dict)):
            if ID == res_dict[k]['id']:
                new_res[kk] = res_dict[k]
    return new_res

def get_stats_for_res(preds,res_dict,test_idxs=None):
    tpreds, taps50, tamaps = [],[],[]
    ttpreds, ttaps50, ttamaps = [],[],[]
    for i in range(len(preds)):
        a = regionprops_table((label(io.imread(preds[i]))))
        napred = len(a['label'])
        aap50 = res_dict[i]['ap'][0]
        amap = res_dict[i]['ap'][0:9].mean()
        if test_idxs:
            if i < len(test_idxs):
                tpreds.append(napred)
                taps50.append(aap50)
                tamaps.append(amap)
            else:
                ttpreds.append(napred)
                ttaps50.append(aap50)
                ttamaps.append(amap)
        else:
            ttpreds.append(napred)
            ttaps50.append(aap50)
            ttamaps.append(amap)
    res_stats = [np.sum(tpreds),np.mean(taps50),np.std(taps50),np.mean(tamaps),np.std(tamaps),
                  np.sum(ttpreds),np.mean(ttaps50),np.std(ttaps50),np.mean(ttamaps),np.std(ttamaps)]
    return res_stats

def get_stats_for_run(pred_list,res_list,titles,p_string_list,labels,test_idxs_list=None):
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