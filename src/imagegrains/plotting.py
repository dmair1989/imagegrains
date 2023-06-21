import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from skimage.measure import label, regionprops
from skimage.segmentation import mark_boundaries 
from skimage.color import label2rgb
from cellpose import io

from imagegrains import grainsizing

def show_training_set(inp_list,mask_str='_mask'):
    for k,f in enumerate(inp_list):
        img = io.imread(str(f))
        plt.subplot(2,len(inp_list),k+1)
    
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(2,len(inp_list),len(inp_list) + k+1)
        seg = io.imread(os.path.splitext(str(f))[0] + mask_str+'.tif')
        #masks= seg['masks'].squeeze()
        plt.imshow(seg)
        plt.axis('off')
    return

def eval_plot(img,y_pred,y_true,j_score,f1,ap,_print=False,title_id =''):
    plt.imshow(mark_boundaries(img, y_pred,mode='thick'))
    plt.imshow(np.ma.masked_where(y_true==0,y_true),alpha=.5)

    t = ["jaccard score (pixel-level): %s " %str(np.round(j_score,decimals=2)),
        'f1 score (pixel-level): %s ' %str(np.round(f1,decimals=2)),
        'AP @50 IoU: %s ' %str(np.round(ap[0], decimals=2)),
        'AP @80 IoU: %s ' %str(np.round(ap[5],decimals=2))
        ]
    
    plt.text(25, 75, t[0], bbox={'facecolor': 'white','edgecolor':'None'},fontsize=15,wrap=True)
    plt.text(25, 145, t[1], bbox={'facecolor': 'white','edgecolor':'None'},fontsize=15,wrap=True)
    plt.text(25, 220, t[2], bbox={'facecolor': 'white','edgecolor':'None'},fontsize=15,wrap=True)
    plt.text(25, 290, t[3], bbox={'facecolor': 'white','edgecolor':'None'},fontsize=15,wrap=True)
    plt.title(title_id)
    if _print == True:
        print('jaccard score (pixel-level): ',np.round(j_score,decimals=2))
        print('f1 score (pixel-level): ',np.round(f1,decimals=2))
        print('AP @50 IoU: ',np.round(ap[0],decimals=2))
        print('AP @80 IoU: ',np.round(ap[5],decimals=2))
    plt.axis('off')
    return

def AP_IoU_plot(eval_results,labels=True,
                thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],title='',test_idxs=None):    
    res_l= [[] for x in range(len(thresholds))]
    for i  in range(len(eval_results)):
        for j in range(len(thresholds)):
            o = eval_results[i]['ap'][j]
            res_l[j].append(o)
    avg_l,std_ul,std_ll =[],[],[]
    for m in range(len(res_l)):
        avg_l.append(np.mean(res_l[m]))
        std_ul.append(np.mean(res_l[m])+np.std(res_l[m]))
        std_ll.append(np.mean(res_l[m])-np.std(res_l[m]))
    
    for i  in range(len(eval_results)):
        if test_idxs:
            if i in test_idxs and i == 0:
                plt.plot(thresholds,eval_results[i]['ap'],'b',label='Test image')
            elif i in test_idxs:
                plt.plot(thresholds,eval_results[i]['ap'],'b')
            elif i == len(eval_results)-1:
                plt.plot(thresholds,eval_results[i]['ap'],'c',alpha=.4,label='Training image')
            else:
                plt.plot(thresholds,eval_results[i]['ap'],'c',alpha=.4)
        elif not test_idxs and i ==0:
            plt.plot(thresholds,eval_results[i]['ap'],'c',alpha=.4,label='Single image')
        else:
            plt.plot(thresholds,eval_results[i]['ap'],'c',alpha=.4)

    plt.plot(thresholds,avg_l,'r',lw=2,label='Dataset avg.')
    plt.fill_between(thresholds,std_ul,std_ll,color='r',alpha=0.2,label='1 Std. dev.')
    plt.xlim(np.min(thresholds),np.max(thresholds))
    plt.ylim(0,1)
    if labels == True:
        plt.ylabel('Average precision (AP)')
        plt.xlabel('IoU threshold')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return

def AP_IoU_summary_plot(eval_results_list,elements,test_idx_list =None ,labels=True,
                        thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]):    
    """
    eval_results_list: list of eval_results from eval_results_list
    elements: dict with the elements to be plotted: 
        dataset = str with the name of the dataset
        model_id = list with the identifier string for predictions from different models        
        colors = list of colors for the different models
        images = bool, if True, plots the AP for each image
        std = bool, if True, plots the standard deviation of the AP for each image
        avg = bool, if True, plots the average AP for the dataset 
    """
    
    for ds in range(len(eval_results_list)):
        res_l= [[] for x in range(len(thresholds))]
        for i  in range(len(eval_results_list[ds])):
            if test_idx_list:
                if i in test_idx_list:
                    for j in range(len(thresholds)):
                        o = eval_results_list[ds][i]['ap'][j]
                        res_l[j].append(o)
                else:
                    continue
            else:
                for j in range(len(thresholds)):
                        o = eval_results_list[ds][i]['ap'][j]
                        res_l[j].append(o)
        avg_l,std_ul,std_ll =[],[],[]
        for m in range(len(res_l)):
            avg_l.append(np.mean(res_l[m]))
            std_ul.append(np.mean(res_l[m])+np.std(res_l[m]))
            std_ll.append(np.mean(res_l[m])-np.std(res_l[m]))
        if not elements['colors']:
            cmap = plt.cm.get_cmap('tab20', len(elements['model_id']))
            elements['colors'] = [cmap(x) for x in range(len(elements['model_id']))]
        if elements['images']==True:
            for i  in range(len(eval_results_list[ds])):
                if i ==0:
                    plt.plot(thresholds,eval_results_list[ds][i]['ap'],'k',alpha=.1,label='Single image')
                else:
                    plt.plot(thresholds,eval_results_list[ds][i]['ap'],color=elements['colors'][ds],alpha=.5)
        if elements['std']==True:
            plt.fill_between(thresholds,std_ul,std_ll,color=elements['colors'][ds],alpha=0.2)
        if elements['avg_model']==True:
            plt.plot(thresholds,avg_l,color=elements['colors'][ds],lw=1.5,label=str(elements['model_id'][ds]))
    plt.xlim(np.min(thresholds),np.max(thresholds))
    plt.ylim(0,1)
    plt.title(str(elements['dataset']))
    if labels == True:
        plt.ylabel('Average precision (AP)')
        plt.xlabel('IoU threshold')
    plt.legend()
    plt.tight_layout()
    return
    
def inspect_predictions(imgs,preds,lbls=None,title='',tar_dir='',save_fig=False):
    """
    Plot images and predictions side by side.  
    `imgs` list of image paths  
    `preds` list of prediction paths
    `lbls` list of label paths (optional)
    `title` title of plot (optional)
    `tar_dir` path to save plot (optional)
    """
    
    if lbls:
        fig = plt.figure(figsize=(len(imgs)*2,len(imgs)), dpi=300)
        rows = 3
    else:
        fig = plt.figure(figsize=(len(imgs)*2,len(imgs)*0.66), dpi=300)
        rows = 2
    for k in range(len(imgs)):
        img = io.imread(str(imgs[k]))
        plt.subplot(rows,len(imgs),k+1)
        plt.imshow(img)
        
        if k == 0:
            plt.ylabel('Image')
        plt.xticks([],[])
        plt.yticks([],[])
        

        plt.subplot(rows,len(imgs),len(imgs) + k+1)
        pred = io.imread(str(preds[k]))
        colors = mask_cmap(pred)
        plt.imshow(label2rgb(pred, image=img, colors=colors, bg_label=0))
        if k == 0:
            plt.ylabel('Predictions')
        plt.xticks([],[])
        plt.yticks([],[])

        if lbls:
            plt.subplot(rows,len(imgs),(len(imgs)*2)+ k+1)
            lbl = io.imread(str(lbls[k]))
            colors = mask_cmap(lbl)
            plt.imshow(label2rgb(lbl, image=img, colors=colors, bg_label=0))
            if k == 0:
                plt.ylabel('Ground truth')
            plt.xticks([],[])
            plt.yticks([],[])
            i_id = Path(imgs[k]).stem
            #i_id = imgs[k].split('\\')[len(imgs[k].split('\\'))-1].split('.')[0]
            plt.xlabel(i_id)      
    if title != '':
        if isinstance(title, list):
            plt.title(title[k])
        else:
            plt.suptitle(title)
    if tar_dir != '' and save_fig == True:
        plt.savefig(tar_dir+'pred_overview.pdf',dpi=300)
    plt.tight_layout()
    return fig

def inspect_dataset_grains(imgs,masks,res_props=None,elements=['image','mask','ellipse_b','ellipse_a','ellipse']):
    fig = plt.subplots(figsize=(18,len(masks)*1.3))
    if not res_props:
        res_props = []
        print('No regionprops: Finding grains...')
        for x in range(len(masks)):
            masks_ = io.imread(str(masks[x]))
            res_props_i = regionprops(label(masks_))
            res_props.append(res_props_i)
    for k in range(len(masks)):
        if not res_props:
            masks_ = io.imread(str(masks))
        m_id = Path(masks[k]).stem
        #m_id = masks[k].split('\\')[len(masks[k].split('\\'))-1].split('.')[0]
        plt.subplot(int(int(np.round(len(masks)/4))), 4, k+1)
        all_grains_plot(io.imread(str((masks[k]))),elements,props=res_props[k],image=io.imread(str(imgs[k])),title=m_id)
        plt.tight_layout()
    return fig

def mask_cmap(masks):
    values = np.unique(masks)
    colors = plt.cm.get_cmap('winter',len(values))
    colors=[colors(i) for i in range(len(values))]
    np.random.shuffle(colors)
    return colors

def show_masks_set(masks,images,show_ap50=False,showmap=False,res_dict=None,show_id=False,title_str=''):
    plt.figure(figsize=(20,2.222*(len(images)/9)))
    if len(images) > 81:
        print('Too many images to show. Showing first 81.')
        images = images[0:80]
        masks = masks[0:80]
    for k in range(len(images)):
        img = io.imread(str(images[k]))
        lbl = io.imread(str(masks[k]))
        colors = mask_cmap(lbl)
        rows = int(len(images)/9)
        rows = 1 if rows == 0 else rows
        plt.subplot(rows,9,k+1)
        
        msks = label2rgb(label(lbl), image=img, bg_label=0,colors=colors)
        plt.imshow(mark_boundaries(msks, label(lbl), color=(1,0,0), mode='thick'))
        plt.axis('off')
        if res_dict != None:
            ap50 = str(np.round(res_dict[k]['ap'][0],decimals=2))
            mAP50_90 = np.mean(res_dict[k]['ap'][0:9])
            if show_ap50 == True:
                plt.title('AP: '+str(ap50))
            elif showmap == True:
                plt.title('mAP: '+str(np.round(mAP50_90,decimals=2)))
        elif show_id == True:
            img_id = Path(images[k]).stem
            plt.title(img_id)
        elif title_str != '':
            if isinstance(title_str, list):
                plt.title(title_str[k])
            else:
                plt.title(title_str)
    plt.tight_layout()
    return

def plot_single_img_pred(image,mask,file_id=None, show_n=False, save=False, tar_dir='',show=False):
    if file_id == None:
        file_id = Path(img).stem
    else:
        file_id = file_id
    img = io.imread(image)
    lbl = io.imread(mask)
    if save == True:
        if tar_dir != '':
            out_dir = f'{tar_dir}/prediction_masks/' 
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = f'{str(Path(image).parent)}/prediction_masks/'
            os.makedirs(out_dir, exist_ok=True)
        tar_dir = out_dir
    with plt.ioff():
        colors = mask_cmap(lbl)
        masks = label2rgb(label(lbl), image=img, bg_label=0,colors=colors)
        plt.imshow(mark_boundaries(masks, label(lbl), color=(1,0,0), mode='thick'))
        plt.axis('off')
        if show_n == True and file_id:
            n = np.unique(label(lbl))
            plt.title(f'{file_id} (n={len(n)})')
        elif file_id:
            plt.title(f'{file_id}')
        elif show_n == True:
            n = np.unique(label(lbl))
            plt.title(f'n={len(n)}')
        plt.tight_layout()
        if save == True:
            plt.savefig(f'{tar_dir}/{file_id}_seg_overlay.png',dpi=300) 

def save_pred_overlays(imgs,preds,save=True,show_n=False,mute=False,tar_dir='',show=False):
    if mute == False:
        for img, pred in tqdm(zip(imgs,preds),desc='Saving images with masks',unit=' images',position=0,leave=True):
            plot_single_img_pred(img,pred,file_id='',show_n=show_n,save=save,tar_dir=tar_dir,show=show)
    else:
        for img, pred in zip(imgs,preds):
            plot_single_img_pred(img,pred,file_id='',show_n=False,save=True,tar_dir=tar_dir,show=show)

def all_grains_plot(masks,elements,props=None, image =None, 
                    fit_res =None,fit_method ='convex_hull',do_fit= False,
                    padding_size=2,title='',plot_padding=15):
    if 'image' in elements and image.any:
        plt.imshow(image)
        h,w,_ = image.shape
        plt.xlim(0-int(w/plot_padding),w+int(w/plot_padding))
        plt.ylim(0-int(h/plot_padding),h+int(h/plot_padding))
    if 'masks_individual' in elements:
        plt.imshow(label2rgb(label(masks), bg_label=0),alpha=0.3)
    if not props:
        print('No regionprops found: Finding grains...')
        props = regionprops(label(masks))
    if not fit_res and do_fit == True:
        print('Fitted axes not found: Attempting axes fit for',fit_method,'...')
        _,_,a_coords,b_coords = grainsizing.fit_grain_axes(props,method=fit_method,padding_size=padding_size)
    if fit_res:
        _,_,a_coords,b_coords = fit_res[0],fit_res[1],fit_res[2],fit_res[3]
        fit_res=[_,_,a_coords,b_coords]
    else:
        a_coords,b_coords = [],[]
    for _idx in range(len(props)):
        miny, minx, maxy, maxx = props[_idx].bbox
        if 'mask' in elements:
            mask = props[_idx].image
            plt.imshow(np.ma.masked_where(mask==0,mask),extent=[minx,maxx,maxy,miny],alpha= .5)
        if 'masks_outline' in elements:
            img_pad = grainsizing.image_padding(props[_idx].image,padding_size=padding_size)
            contours = grainsizing.contour_grain(img_pad)
            for contour in contours:
                plt.plot(contour[:, 1]-(padding_size-.5)+minx, contour[:, 0]-(padding_size-.5)+miny,'-c',linewidth=1.5)
        if 'convex_hull' in elements:
            convex_image = props[_idx].convex_image
            conv_pad = grainsizing.image_padding(convex_image,padding_size=padding_size)
            #conv_pad = cv.copyMakeBorder(convex_image.astype(int), padding_size, padding_size, padding_size, padding_size, cv.BORDER_CONSTANT, (0,0,0))
            contours = grainsizing.contour_grain(conv_pad)
            for contour in contours:
                plt.plot(contour[:, 1]-(padding_size-.5)+minx, contour[:, 0]-(padding_size-.5)+miny,'-m',linewidth=1.5)
        if 'fit_a' in elements and a_coords:
            x = [a_coords[_idx][0][1]-(padding_size-.5)+minx,a_coords[_idx][1][1]-(padding_size-.5)+minx]
            y = [a_coords[_idx][0][0]-(padding_size-.5)+miny,a_coords[_idx][1][0]-(padding_size-.5)+miny]
            plt.plot(x,y,'r',label='a-xis[convex hull]',) 
        if 'fit_b' in elements and b_coords:
            if not b_coords[_idx]:
                print(_idx,': b-axis error')
            else:
                x1 = [b_coords[_idx][0][1]-(padding_size-.5)+minx,b_coords[_idx][1][1]-(padding_size-.5)+minx]
                y1 = [b_coords[_idx][0][0]-(padding_size-.5)+miny,b_coords[_idx][1][0]-(padding_size-.5)+miny]
            plt.plot(x1,y1,'b',label='b_axis[convex hull]',)   
        if 'ellipse' in elements:    
            x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= ell_from_props(props,_idx)
            plt.plot(x, y, 'r--', linewidth=1.5)
        if 'ellipse_b' in elements:
            x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= ell_from_props(props,_idx)
            plt.plot((x1, x4), (y1, y4), '-b', linewidth=1)
        if 'ellipse_a' in elements:
            x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= ell_from_props(props,_idx)
            plt.plot((x2, x3), (y2, y3), '-r', linewidth=1)
        if 'ellipse_center' in elements:
            plt.plot(x0, y0, '.g', markersize=2)       
        if 'bbox' in elements:
            bx = (minx, maxx, maxx, minx, minx)
            by = (miny, miny, maxy, maxy, miny)
            plt.plot(bx, by, '-r', linewidth=1,label='bbox')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.tight_layout()
    return

def plot_single_img_mask(img, mask,file_id):
    colors = mask_cmap(mask)
    masks = label2rgb(label(mask), image=img, bg_label=0,colors=colors)
    plt.imshow(mark_boundaries(masks, label(mask), color=(1,0,0), mode='thick'))
    plt.axis('off')
    plt.title(file_id)

def single_grain_plot(mask,elements,props=None, image =None, fit_res =None,
                        fit_method ='convex_hull',do_fit= False,
                        padding_size=2,figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    if not props:
        print('No regionprops found: Finding grains...')
        props = regionprops(label(mask))
    miny, minx, maxy, maxx = props[0].bbox
    if 'image' in elements and image.any:
        plt.imshow(image,extent=[minx,maxx,maxy,miny])
    if 'masks_individual' in elements:
        plt.imshow(label2rgb(label(mask), bg_label=0),alpha=0.3)
    if not fit_res and do_fit == True:
        print('Fitted axes not found: Attempting axes fit for',fit_method,'...')
        _,_,a_coords,b_coords = grainsizing.fit_grain_axes(props,method=fit_method,padding_size=padding_size)
        fit_res=[_,_,a_coords,b_coords]
    if fit_res:
        _,_,a_coords,b_coords = fit_res[0],fit_res[1],fit_res[2],fit_res[3]
    else:
        a_coords,b_coords = [],[]
    if 'mask' in elements:
        plt.imshow(np.ma.masked_where(mask==0,mask),extent=[minx-padding_size,maxx+padding_size,maxy+padding_size,miny-padding_size],alpha= .5)
    if 'masks_outline' in elements:
        img_pad = grainsizing.image_padding(props[0].image,padding_size=padding_size)
        contours = grainsizing.contour_grain(img_pad)
        for contour in contours:
                plt.plot(contour[:, 1]-(padding_size-.5)+minx, contour[:, 0]-(padding_size-.5)+miny,'-c',linewidth=1.5)
    if 'convex_hull' in elements:
        convex_image = props[0].convex_image
        conv_pad = grainsizing.image_padding(convex_image,padding_size=padding_size)
            #conv_pad = cv.copyMakeBorder(convex_image.astype(int), padding_size, padding_size, padding_size, padding_size, cv.BORDER_CONSTANT, (0,0,0))
        contours = grainsizing.contour_grain(conv_pad)
        for contour in contours:
            plt.plot(contour[:, 1]-(padding_size-.5)+minx, contour[:, 0]-(padding_size-.5)+miny,'-m',linewidth=1)
    if 'fit_a' in elements and a_coords:
        x = [a_coords[0][0][1]-(padding_size-.5)+minx,a_coords[0][1][1]-(padding_size-.5)+minx]
        y = [a_coords[0][0][0]-(padding_size-.5)+miny,a_coords[0][1][0]-(padding_size-.5)+miny]
        plt.plot(x,y,'r',label='a-xis[convex hull]',) 
    if 'fit_b' in elements and b_coords:
        if not b_coords[0]:
            print(0,': b-axis error')
        else:
            x1 = [b_coords[0][0][1]-(padding_size-.5)+minx,b_coords[0][1][1]-(padding_size-.5)+minx]
            y1 = [b_coords[0][0][0]-(padding_size-.5)+miny,b_coords[0][1][0]-(padding_size-.5)+miny]
        plt.plot(x1,y1,'b',label='b_axis[convex hull]',)   
    if 'ellipse' in elements:    
        x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= ell_from_props(props)
        plt.plot(x, y, 'r--', linewidth=1)
    if 'ellipse_b' in elements:
        x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= ell_from_props(props)
        plt.plot((x1, x4), (y1, y4), '-b', linewidth=1)
    if 'ellipse_a' in elements:
        x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= ell_from_props(props)
        plt.plot((x2, x3), (y2, y3), '-r', linewidth=1)
    if 'ellipse_center' in elements:
        plt.plot(x0, y0, '.g', markersize=2)       
    if 'bbox' in elements:
        bx = (minx, maxx, maxx, minx, minx)
        by = (miny, miny, maxy, maxy, miny)
        plt.plot(bx, by, '-r', linewidth=2,label='bbox')
    #plt.axis('off')
    return(fig)

def ell_from_props(props,_idx=0):
    y0, x0 = props[_idx].centroid
    b, a = props[_idx].major_axis_length, props[_idx].minor_axis_length
    orientation = props[_idx].orientation
    x1 = x0 + np.cos(orientation) * .5 * a
    x4 = x0 - np.cos(orientation) * .5 * a
    y1 = y0 - np.sin(orientation) * .5 * a
    y4 = y0 + np.sin(orientation) * .5 * a
    x2 = x0 - np.sin(orientation) * .5 * b
    x3 = x0 + np.sin(orientation) * .5 * b
    y2 = y0 - np.cos(orientation) * .5 * b
    y3 = y0 + np.cos(orientation) * .5 * b 
    phi = np.linspace(0,2*np.pi,50)
    x = x0 + a/2 * np.cos(phi) * np.cos(-orientation) - b/2 * np.sin(phi) * np.sin(-orientation)
    y = y0 + a/2 * np.cos(phi) * np.sin(-orientation) + b/2 * np.sin(phi) * np.cos(-orientation)
    return(x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y)
        
def plot_gsd(gsd,color='c', perc_range=np.arange(0.01,1.01,0.01),length_max=300,gsd_id=None,title=None,label_axes=False,lw=.75,orientation='vertical',units='px',alpha=1):
        if orientation == 'vertical':
            xmax = length_max
            xmin = 0
            ymax= np.max(perc_range)
            ymin = np.min(perc_range)
            y = perc_range
            x = gsd
            if label_axes != False:
                plt.xlabel('Grain Size ( '+str(units)+')')
                plt.ylabel('Fraction smaller')
        elif orientation == 'horizontal':
            ymax = length_max
            ymin = 0
            xmax = np.max(perc_range)
            xmin = np.min(perc_range)
            x = perc_range
            y = gsd 
            if label_axes != False:
                plt.xlabel('Grain Size ( '+str(units)+')')
                plt.ylabel('Fraction smaller')
        if not gsd_id:
                plt.plot(x,y,color=color,linewidth=lw)
        else:
                plt.plot(x,y,color=color,label=gsd_id,linewidth=lw)
        plt.ylim(ymin,ymax)
        plt.xlim(xmin,xmax)

        if title:
                plt.title(title,fontsize=8)

        plt.tight_layout()

def plot_gsd_uncert(uncert_res,perc_range=np.arange(0.01,1.01,0.01),color='k',uncert_area=True,uncert_bounds=False,uncert_median=False,orientation='vertical'):
    uci,lci,med = uncert_res[1],uncert_res[2],uncert_res[0]
    if not any(uci):
         pass
    else:
        if orientation == 'vertical':
            if uncert_area == True:
                plt.fill_betweenx(perc_range,uci,lci,alpha=0.2,color=color)
            if uncert_median == True:
                plt.plot(med,perc_range,color=color,linewidth=1)
            if uncert_bounds == True:
                plt.plot(uci,perc_range,color=color,linewidth=1,linestyle='--')
                plt.plot(lci,color=color,linewidth=1,linestyle='--')

        elif orientation == 'horizontal':
            if uncert_area == True:
                plt.fill_between(perc_range,uci,lci,alpha=0.2,color=color)
            if uncert_median == True:
                plt.plot(perc_range,med,color=color,linewidth=1)
            if uncert_bounds == True:
                plt.plot(perc_range,uci,color=color,linewidth=1,linestyle='--')
                plt.plot(perc_range,lci,color=color,linewidth=1,linestyle='--')

def plot_gsd_deltas(uncert_res,gsd,baseline,perc_range=np.arange(0.01,1.01,0.01),color='k',label_axes=True
                    ,uncert_area=True,uncert_bounds=False,uncert_median=False,orientation='horizontal'):
    uci,lci,med = uncert_res[1],uncert_res[2],uncert_res[0]
    if not any(uci):
         pass
    else:
        gsd_norm = [((baseline[j]-gsd[j])/baseline[j])*100 for j in range(len(baseline))]
        ub = [gsd_norm[j]+(((uci[j]-lci[j])/baseline[j])*100)/2 for j in range(len(gsd_norm))]
        lb = [gsd_norm[j]-(((uci[j]-lci[j])/baseline[j])*100)/2 for j in range(len(gsd_norm))]
        med_norm = [med[j]-baseline[j] for j in range(len(baseline))]
        if orientation == 'vertical':
            x = gsd_norm
            y = perc_range
            xmed = med_norm
            ymed = perc_range
            
            if uncert_area == True:
                plt.fill_betweenx(y,ub,lb,alpha=0.2,color=color)
            if uncert_bounds == True:
                plt.plot(x,ub,color=color,linewidth=1,linestyle='--')
                plt.plot(x,lb,color=color,linewidth=1,linestyle='--')
        elif orientation == 'horizontal':
            x = perc_range
            y = gsd_norm
            xmed = perc_range
            ymed = med_norm

            if uncert_area == True:
                plt.fill_between(x,ub,lb,alpha=0.2,color=color)
            if uncert_bounds == True:
                plt.plot(ub,y,color=color,linewidth=1,linestyle='--')
                plt.plot(lb,y,color=color,linewidth=1,linestyle='--')
        plt.plot(x,y,color=color,linewidth=1)
        if uncert_median == True:
            plt.plot(xmed,ymed,color=color,linewidth=1)
