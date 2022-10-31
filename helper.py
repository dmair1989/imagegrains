from dataclasses import dataclass
import os
import numpy as np
from cellpose import io
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from cellpose import models

class prediction_helper:
    
    def predict_folder(INP_DIR,model,image_format='jpg',channels=[0,0],diameter=None,min_size=15,rescale=None,TAR_DIR='',return_results=False,save_masks=True,mask_l=[],flow_l=[],styles_l=[],ID_l=[],img_l=[]):
        """ Do predictions on all images with `image_format` extension in a folder. 
        If `return_results` is `True` respective lists of 1D arrays for predicted *masks*, *flows* and *styles* 
        from `CellposeModel.eval()` are returned (see https://cellpose.readthedocs.io/en/latest/api.html#id5).

        Parameters:
        ------------
        IND_DIR: Directory path (str)
        model: Trained model from 'models.CellposeModel' class. 
            Use either `models.CellposeModel(model_type='')` for built-in cellpose models or 
            `models.CellposeModel(pretrained_model='') for custom models.
            See https://cellpose.readthedocs.io/en/latest/models.html for more details
        image_format (str(optional, default 'jpg'))
        return_results (bool(optional, default False))
        TAR_DIR (str(optional, default '')) - output directory
        save_masks (bool(optional, default True)) - flag for saving predicted mask as `.tif` files in `TAR_DIR`

        Parameters that can be handed down to`CellposeModel.eval()`, 
        see https://cellpose.readthedocs.io/en/latest/api.html#id5 :

        channels (list (optional, default [0,0]))
        diameter (float (optional, default None))
        rescale (float (optional, default None))
        min_size (int (optional, default 15))   
        
        
        Returns
        ------------
        masks (optional; list of 2D arrays)
            labelled image, where 0=no masks; 1,2,…=mask labels

        flows (optional; list of lists 2D arrays) 
            flows[k][0] = XY flow in HSV 0-255 flows[k][1] = XY flows at each pixel 
            flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics) 
            flows[k][3] = final pixel locations after Euler integration

        styles (optional; list of 1D arrays of length 64) 
            style vector summarizing each image, also used to estimate size of objects in image

        """
        try:
            Z = sorted(glob(INP_DIR+'/*.'+image_format))
            print('Predicting masks for files in',INP_DIR,'...')                
            for j in tqdm(range(len(Z)),unit='image',colour='CYAN'):
                img= io.imread(Z[j])
                ID = Z[0].split('\\')[len(Z[0].split('\\'))-1].split('.')[0]
                if any(x in 'ID' for x in ['flows', 'masks','mask',]):
                    print('')
                else:
                    masks, flows, styles = model.eval(img, diameter=diameter,rescale=rescale,min_size=min_size,channels=channels) 
                    if save_masks == False and return_results == False:
                        print('Saving and returning of results weres switched of - therefore mask saving was turned on!')
                        save_masks = True
                    if save_masks == True:
                        io.imsave(TAR_DIR+ID+'_pred.tif',masks)
                    if return_results == True:
                        mask_l.append(masks),flow_l.append(flows),styles_l.append(styles),ID_l.append(ID)
            print('Sucessfully created predictions for',j+1,'images.')
        except KeyboardInterrupt:
            print('Aborted.')
        if return_results == True:
            img_l = Z
            return(mask_l,flow_l,styles_l,ID_l,img_l)

    def predict_dataset(INP_DIR,model,image_format='jpg',channels=[0,0],diameter=None,min_size=15,rescale=None,TAR_DIR='',return_results=False,save_masks=True):
        dirs = next(os.walk(INP_DIR))[1]
        if 'train' in dirs:
            W_DIR = INP_DIR+'/train'
            mask_l,flow_l,styles_l,ID_l,img_l = prediction_helper.predict_folder(W_DIR,model,image_format=image_format,channels=channels,diameter=diameter,
            min_size=min_size,rescale=rescale,TAR_DIR=TAR_DIR,return_results=return_results,save_masks=save_masks)
        if 'test' in dirs:       
            W_DIR = INP_DIR+'/test'
            if not mask_l:
                mask_l=[],flow_l=[],styles_l=[],ID_l=[],img_l=[]
            mask_l,flow_l,styles_l,ID_l,img_l = prediction_helper.predict_folder(W_DIR,model,image_format=image_format,channels=channels,
            diameter=diameter,min_size=min_size,rescale=rescale,TAR_DIR=TAR_DIR,return_results=return_results,save_masks=save_masks,
            mask_l=mask_l,flow_l=flow_l,styles_l=styles_l,ID_l=ID_l,img_l=img_l)
        else:
            print('No "train" and/or "test" directory found - trying to find images in given directory.')
            mask_l,flow_l,styles_l,ID_l,img_l = prediction_helper.predict_folder(W_DIR,model,image_format=image_format,channels=channels,diameter=diameter,
            min_size=min_size,rescale=rescale,TAR_DIR=TAR_DIR,return_results=return_results,save_masks=save_masks)
        return(mask_l,flow_l,styles_l,ID_l,img_l)
    
    def models_from_zoo(MOD_DIR):
        model_list = natsorted(glob(MOD_DIR+'*.*'))
        try:
            models.CellposeModel(pretrained_model=model_list[0])
        except:
            print('No cellpose model found in this directory.')
        M_ID = [model_list[i].split('\\')[len(model_list[i].split('\\'))-1].split('.')[0] for i in range(len(model_list))]
        return(model_list,M_ID)

    def batch_predict(MOD_DIR,DIR_PATHS,configuration=[],image_format='jpg',channels=[0,0],diameter=None,min_size=15,
    rescale=None,TAR_DIR='',return_results=False,save_masks=True):
        model_list,M_ID = prediction_helper.models_from_zoo(MOD_DIR)
        for m_idx in range(len(model_list)):
            model = models.CellposeModel(pretrained_model=model_list[m_idx])
            mID = M_ID[m_idx]
            print(mID,'found...')
            if configuration[m_idx]:
                for key,val in configuration[m_idx].items():
                    exec(key + '=val')
                print('... with custom configuration.')
            for d_idx in range(len(DIR_PATHS)):
                if return_results == True:
                    all_results = {}
                mask_l,flow_l,styles_l,ID_l,img_l = prediction_helper.predict_dataset(DIR_PATHS[d_idx],model,
                image_format=image_format,channels=channels,diameter=diameter,min_size=min_size,rescale=rescale,TAR_DIR=TAR_DIR,
                return_results=return_results,save_masks=save_masks)
                if return_results == True:
                    dataset_res = {'masks':mask_l,'flows':flow_l,'styles':styles_l,'id':ID_l,'images':img_l}
                    all_results[str(mID)+'_'+str(d_idx)]=dataset_res
        return(all_results)

    def images_from_split_data(IMG_DIR):
        img_list=[]
        #try if train, test directory exist
            #get files from there
        #else: get files from IMG_DIR
        #filter for png, tif, jpg
        #remove imgs with _mask, _masks, _flows strings
        return(img_list)
    
    def labels_from_split_data(LBL_DIR):
        lbl_list = []
        #try if train, test directory exist
            #get files from there
        #else: get files from IMG_DIR
        #filter for png, tif, jpg
        #remove entries that have not _mask, _masks strings
        return(lbl_list)
