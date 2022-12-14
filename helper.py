import os,pickle
from cellpose import io
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from skimage.measure import label

from cellpose import metrics
from cellpose import models

from GrainSizing import filter

class train:

    def load_data(PATH,mask_str='mask',im_str='',im_format='jpg',mask_format='tif'):
        dirs = next(os.walk(PATH))[1]
        W_PATH = []
        if 'test' in dirs:
            W_PATH = str(PATH+'/test/')
            test_images = train.load_imgs_masks(W_PATH,format=im_format,filter_str=im_str)
            test_masks = train.load_imgs_masks(W_PATH,format=mask_format,filter_str=mask_str)
        if 'train' in dirs:
            W_PATH = str(PATH+'/train/')
            train_images = train.load_imgs_masks(W_PATH,format=im_format,filter_str=im_str)
            train_masks = train.load_imgs_masks(W_PATH,format=mask_format,filter_str=mask_str)
        if not W_PATH:
            W_PATH = PATH
            train_images = train.load_imgs_masks(W_PATH,format=im_format,filter_str=im_str)
            train_masks = train.load_imgs_masks(W_PATH,format=mask_format,filter_str=mask_str)
            test_images,test_masks = [],[]
        return(train_images,train_masks,test_images,test_masks)

    def load_imgs_masks(W_PATH,format='',filter_str=''):
        ret_list = natsorted(glob(W_PATH+'/*'+filter_str+'*.'+format))
        return(ret_list)
    
    def check_labels(labels,TAR_DIR='',lbl_str='_mask',mask_format='tif'):
        track_l = []
        for i in labels:
            if lbl_str in i:
                continue
            else:
                img= io.imread(i)
                ID = i.split('\\')[len(i.split('\\'))-1].split('.')[0]
                print(ID)
                #plt.imshow(img)
                io.imsave(TAR_DIR+'/'+ID+lbl_str+'.'+mask_format,img)
                track_l.append(ID)
        if len(track_l) == 0:
            print('No files renamed.')
        return(track_l)

    def check_im_label_pairs(img_list,lbl_list):
        error_list=[]
        for k in img_list:
            ID = k.split('\\')[len(k.split('\\'))-1].split('.')[0]
            if any(ID in x for x in lbl_list):
                continue
            else:
                error_list.append(k)
        if len(error_list)==0:
            print('All images have labels.')
        return(error_list)

class prediction:
    
    def predict_folder(INP_DIR,model,image_format='jpg',filter_str='',channels=[0,0],diameter=None,min_size=15,rescale=None,TAR_DIR='',
    return_results=False,save_masks=True,mute=False,mID=''):
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
        filter_str (str(optional, default ''))
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
        mask_l,flow_l,styles_l,ID_l,img_l = [],[],[],[],[]
        try:
            Z = natsorted(glob(INP_DIR+'/*'+filter_str+'*.'+image_format))
            if mute== False:
                print('Predicting for ',INP_DIR,'...')
            count=0                
            for j in tqdm(range(len(Z)), desc=INP_DIR,unit='image',colour='CYAN'):
                img= io.imread(Z[j])
                ID = Z[j].split('\\')[len(Z[j].split('\\'))-1].split('.')[0]
                if any(x in ID for x in ['flow','flows','masks','mask']):
                    continue
                else:
                    masks, flows, styles = model.eval(img, diameter=diameter,rescale=rescale,min_size=min_size,channels=channels) 
                    if save_masks == False and return_results == False:
                        print('Saving and returning of results weres switched of - therefore mask saving was turned on!')
                        save_masks = True
                    if save_masks == True:
                        if TAR_DIR:
                            try:
                                os.makedirs(TAR_DIR)
                            except FileExistsError:
                                pass
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
        return(mask_l,flow_l,styles_l,ID_l,img_l)

    def predict_dataset(INP_DIR,model,image_format='jpg',channels=[0,0],diameter=None,min_size=15,rescale=None,TAR_DIR='',
    return_results=False,save_masks=True,mute=False,do_subfolders=False,mID=''):
        mask_ll,flow_ll,styles_ll,ID_ll,=[],[],[],[]
        dirs = next(os.walk(INP_DIR))[1]
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
                mask_l_i,flow_l_i,styles_l_i,ID_l_i,_ = prediction.predict_folder(W_DIR,model,image_format=image_format,channels=channels,diameter=diameter,
                min_size=min_size,rescale=rescale,TAR_DIR=TAR_DIR,return_results=return_results,save_masks=save_masks,mute=mute,mID=mID)
                if return_results==True:
                    for x in range(len(mask_l_i)):
                        mask_ll.append(mask_l_i[x])
                        flow_ll.append(flow_l_i[x])
                        styles_ll.append(styles_l_i[x])
                        ID_ll.append(ID_l_i[x])
            else:
                continue
        return(mask_ll,flow_ll,styles_ll,ID_ll)
    
    def models_from_zoo(MOD_DIR):
        model_list = natsorted(glob(MOD_DIR+'/*.*'))
        try:
            models.CellposeModel(gpu=True,pretrained_model=model_list[0])
        except:
            print('No cellpose model found in this directory.')
        M_ID = [model_list[i].split('\\')[len(model_list[i].split('\\'))-1].split('.')[0] for i in range(len(model_list))]
        return(model_list,M_ID)

    def batch_predict(MOD_DIR,DIR_PATHS,configuration=[],image_format='jpg',channels=[0,0],diameter=None,min_size=15,
    rescale=None,TAR_DIR='',return_results=False,save_masks=True,mute=False,do_subfolders=False):
        model_list,M_ID = prediction.models_from_zoo(MOD_DIR)
        all_results= {}
        for m_idx in range(len(model_list)):
            model = models.CellposeModel(gpu=True,pretrained_model=model_list[m_idx])
            mID = M_ID[m_idx]
            print(mID,'found...')
            if configuration:
                if len(configuration)>1:
                    try:
                        for key,val in configuration[m_idx].items():
                            exec(key + '=val')
                        if mute == False:
                            print('... with custom configuration.')
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
            for d_idx in range(len(DIR_PATHS)):
                all_mask,all_flow_l,all_styles_l,all_ID_l = prediction.predict_dataset(DIR_PATHS[d_idx],model,
                image_format=image_format,channels=channels,diameter=diameter,min_size=min_size,rescale=rescale,TAR_DIR=TAR_DIR,
                return_results=return_results,save_masks=save_masks,mute=mute,do_subfolders=do_subfolders,mID=mID)
                if return_results == True:
                    dataset_res = {'masks':all_mask,'flows':all_flow_l,'styles':all_styles_l,'id':all_ID_l}
                    all_results[str(mID)+'_'+str(d_idx)]=dataset_res
        return(all_results)

class eval:

    def dataset_loader(IM_DIRs,image_format='jpg',label_format='tif',pred_format='tif',label_str='',pred_str=''):
        imgs,lbls,preds = [],[],[]
        dirs = next(os.walk(IM_DIRs[0]))[1]
        IM_DIR = []
        if 'test' in dirs:
                IM_DIR += [str(IM_DIRs[0]+'/test/')]
        if 'train' in dirs:
                IM_DIR += [str(IM_DIRs[0]+'/train/')]
        if not IM_DIR:
            IM_DIR = IM_DIRs
        for i in range(len(IM_DIR)):
            imgs1,lbls1,preds1 = eval.load_from_folders(IM_DIR[i],image_format=image_format,label_format=label_format,pred_format=pred_format,label_str=label_str,pred_str=pred_str)
            imgs += imgs1
            lbls += lbls1
            preds += preds1
        return(imgs,lbls,preds)

    def load_from_folders(IM_DIR,LBL_DIR='',PRED_DIR='',image_format='jpg',label_format='tif',pred_format='tif',label_str='',pred_str=''):
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
        return(imgs,lbls,preds)

    def eval_image(y_true,y_pred,thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]):
        ap, tp, fp, fn = metrics.average_precision(label(y_true),label(y_pred),threshold=thresholds)
        iout, preds = metrics.mask_ious(label(y_true),label(y_pred))
        #j_score = jaccard_score(y_true, y_pred,average="macro")
        #f1 = f1_score(y_true,y_pred,average="macro")
        return(ap, tp, fp, fn, iout, preds)
    
    def eval_set(imgs,lbls,preds,dataID='',TAR_DIR='',thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
        filters={'edge':[False,.05],'px_cutoff':[False,10]},filter_props=['label','area','centroid','major_axis_length','minor_axis_length'],
        save_results=True,return_results=True):  
        eval_results={}
        for i in range(len(imgs)):
            img = io.imread(imgs[i])
            y_true = io.imread(lbls[i])
            y_pred = io.imread(preds[i])

            if filters:
                _, y_true = filter.filter_grains(labels=y_true,properties=filter_props,filters=filters,mask=y_true)
                _, y_pred = filter.filter_grains(labels=y_pred,properties=filter_props,filters=filters,mask=y_pred)
            ap,_,_,_,iout,_ =  eval.eval_image(y_true,y_pred, thresholds=thresholds)

            eval_results[i] = {'img':img, 'ap':ap, 'iout':iout,}
        if save_results==True:
            if TAR_DIR:
                try:
                    os.makedirs(TAR_DIR)
                except FileExistsError:
                    pass
                export = TAR_DIR+'/'+dataID+'_eval_res.pkl'
            else:
                export = dataID+'_eval_res.pkl'
            with open(str(export), 'wb') as f:
                pickle.dump(eval_results, f)
        if return_results == True:
            return(eval_results)

    def load_eval_res(name,PATH=''):
        with open(PATH +'/'+ name +'.pkl', 'rb') as f:
            eval_results = pickle.load(f)
        return(eval_results)