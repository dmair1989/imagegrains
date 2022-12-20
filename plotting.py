import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.segmentation import mark_boundaries 
from skimage.color import label2rgb
from cellpose import io

from GrainSizing import measure

class training:

    def show_training_set(inp_list,mask_str='_mask'):
        for k,f in enumerate(inp_list):
            img = io.imread(f)
            plt.subplot(2,len(inp_list),k+1)
        
            plt.imshow(img)
            plt.axis('off')

            plt.subplot(2,len(inp_list),len(inp_list) + k+1)
            seg = io.imread(os.path.splitext(f)[0] + mask_str+'.tif')
            #masks= seg['masks'].squeeze()
            plt.imshow(seg)
            plt.axis('off')

class segmentation:

    def eval_plot(img,y_pred,y_true,j_score,f1,ap,_print=False,ID =''):
     fig = plt.figure(figsize=(15, 15))
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
     plt.title(ID)
     if _print == True:
          print('jaccard score (pixel-level): ',np.round(j_score,decimals=2))
          print('f1 score (pixel-level): ',np.round(f1,decimals=2))
          print('AP @50 IoU: ',np.round(ap[0],decimals=2))
          print('AP @80 IoU: ',np.round(ap[5],decimals=2))
     plt.axis('off')
     return(fig)
    
    def AP_IoU_plot(eval_results,thresholds=[0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],title=''):    
        res_l= [[] for x in range(len(thresholds))]
        for i  in range(len(eval_results)):
            for j in range(len(thresholds)):
                o = eval_results[i]['ap'][j]
                res_l[j].append(o)
        avg_l,std_ul,std_ll =[],[],[]
        plt.figure(figsize=(3, 5))
        for m in range(len(res_l)):
            avg_l.append(np.mean(res_l[m]))
            std_ul.append(np.mean(res_l[m])+np.std(res_l[m]))
            std_ll.append(np.mean(res_l[m])-np.std(res_l[m]))
        plt.fill_between(thresholds,std_ul,std_ll,color='r',alpha=0.2,label='SD of dataset')
        for i  in range(len(eval_results)):
            if i ==0:
                plt.plot(thresholds,eval_results[i]['ap'],'k',alpha=.5,label='Single image')
            else:
                plt.plot(thresholds,eval_results[i]['ap'],'k',alpha=.5)
        plt.plot(thresholds,avg_l,'r',lw=2,label='Dataset mean')
        plt.xlim(np.min(thresholds),np.max(thresholds))
        plt.ylim(0,1)
        plt.ylabel('Average precision (AP)')
        plt.xlabel('IoU threshold')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        return()
    
    def AP_IoU_summary_plot(eval_results_list,elements,thresholds=[0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]):    
        fig = plt.figure(figsize=(3, 5))
        for ds in range(len(eval_results_list)):
            res_l= [[] for x in range(len(thresholds))]
            for i  in range(len(eval_results_list[ds])):
                for j in range(len(thresholds)):
                    o = eval_results_list[ds][i]['ap'][j]
                    res_l[j].append(o)
            avg_l,std_ul,std_ll =[],[],[]
            for m in range(len(res_l)):
                avg_l.append(np.mean(res_l[m]))
                std_ul.append(np.mean(res_l[m])+np.std(res_l[m]))
                std_ll.append(np.mean(res_l[m])-np.std(res_l[m]))
            if not elements['colors']:
                cmap = plt.cm.get_cmap('tab20', len(elements['model_ID']))
                elements['colors'] = [cmap(x) for x in range(len(elements['model_ID']))]
            if elements['images']==True:
                for i  in range(len(eval_results_list[ds])):
                    if i ==0:
                        plt.plot(thresholds,eval_results_list[ds][i]['ap'],'k',alpha=.1,label='Single image')
                    else:
                        plt.plot(thresholds,eval_results_list[ds][i]['ap'],color=elements['colors'][ds],alpha=.5)
            if elements['SD']==True:
                plt.fill_between(thresholds,std_ul,std_ll,color=elements['colors'][ds],alpha=0.2)
            if elements['avg_model']==True:
                plt.plot(thresholds,avg_l,color=elements['colors'][ds],lw=2,label=str(elements['model_ID'][ds]))
        plt.xlim(np.min(thresholds),np.max(thresholds))
        plt.ylim(0,1)
        plt.title(str(elements['dataset']))
        plt.ylabel('Average precision (AP)')
        plt.xlabel('IoU threshold')
        plt.legend()
        plt.tight_layout()
        return(fig)
    
    def inspect_predictions(imgs,lbls,preds,PATH=''):
        fig = plt.figure(figsize=(len(imgs)*2,7), dpi=300)
        for k in range(len(imgs)):
            img = io.imread(imgs[k])
            plt.subplot(3,len(imgs),k+1)
            plt.imshow(img)
            
            if k == 0:
                plt.ylabel('Image')
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            

            plt.subplot(3,len(imgs),len(imgs) + k+1)
            lbl = io.imread(lbls[k])
            plt.imshow(label2rgb(lbl, image=img, bg_label=0))
            if k == 0:
                plt.ylabel('Ground truth')
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()

            plt.subplot(3,len(imgs),(len(imgs)*2)+ k+1)
            prd = io.imread(preds[k])
            plt.imshow(label2rgb(prd, image=img, bg_label=0))
            if k == 0:
                plt.ylabel('Predictions')
            plt.xticks([],[])
            plt.yticks([],[])
            i_ID = imgs[k].split('\\')[len(imgs[k].split('\\'))-1].split('.')[0]
            plt.xlabel(i_ID)
            plt.tight_layout()
        plt.suptitle(str(PATH))
        return(fig)
    
class grains:

    def inspect_dataset_grains(imgs,masks,res_props=None,elements=['image','mask','ellipse_b','ellipse_a','ellipse']):
        fig = plt.subplots(figsize=(18,len(masks)*1.3))
        if not res_props:
            res_props = []
            print('No regionprops: Finding grains...')
            for x in range(len(masks)):
                masks_ = io.imread(masks[x])
                res_props_i = regionprops(label(masks_))
                res_props.append(res_props_i)
        for k in range(len(masks)):
            if not res_props:
                masks_ = io.imread(masks)
            m_ID = masks[k].split('\\')[len(masks[k].split('\\'))-1].split('.')[0]
            plt.subplot(np.int(np.int(np.round(len(masks)/4))), 4, k+1)
            grains.all_grains_plot(io.imread(masks[k]),elements,props=res_props[k],image=io.imread(imgs[k]),title=m_ID)
            plt.tight_layout()
        return(fig)

    def all_grains_plot(masks,elements,props=None, image =None, 
                        fit_res =None,fit_method ='convex_hull',do_fit= False,
                        padding_size=2,title='',plot_padding=15):
        if 'image' in elements and image.any:
            plt.imshow(image)
            h,w,_ = image.shape
            plt.xlim(0-np.int(w/plot_padding),w+np.int(w/plot_padding))
            plt.ylim(0-np.int(h/plot_padding),h+np.int(h/plot_padding))
        if 'masks_individual' in elements:
            plt.imshow(label2rgb(label(masks), bg_label=0),alpha=0.3)
        if not props:
            print('No regionprops found: Finding grains...')
            props = regionprops(label(masks))
        if not fit_res and do_fit == True:
            print('Fitted axes not found: Attempting axes fit for',fit_method,'...')
            _,_,a_coords,b_coords = measure.fit_grain_axes(props,method=fit_method,padding_size=padding_size)
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
                img_pad = measure.image_padding(props[_idx].image,padding_size=padding_size)
                contours = measure.contour_grain(img_pad)
                for contour in contours:
                    plt.plot(contour[:, 1]-(padding_size-.5)+minx, contour[:, 0]-(padding_size-.5)+miny,'-c',linewidth=1.5)
            if 'convex_hull' in elements:
                convex_image = props[_idx].convex_image
                conv_pad = measure.image_padding(convex_image,padding_size=padding_size)
                #conv_pad = cv.copyMakeBorder(convex_image.astype(int), padding_size, padding_size, padding_size, padding_size, cv.BORDER_CONSTANT, (0,0,0))
                contours = measure.contour_grain(conv_pad)
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
                x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= grains.ell_from_props(props,_idx)
                plt.plot(x, y, 'r--', linewidth=1.5)
            if 'ellipse_b' in elements:
                x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= grains.ell_from_props(props,_idx)
                plt.plot((x1, x4), (y1, y4), '-b', linewidth=1)
            if 'ellipse_a' in elements:
                x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= grains.ell_from_props(props,_idx)
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
        return()

    def single_grain_plot(mask,elements,props=None, image =None, fit_res =None,
                            fit_method ='convex_hull',do_fit= False,
                            padding_size=2,figsize=(8,8)):
        fig = plt.figure(figsize=figsize)
        if not props:
            print('No regionprops found: Finding grains...')
            props = regionprops(label(grains))(mask)
        miny, minx, maxy, maxx = props[0].bbox
        if 'image' in elements and image.any:
            plt.imshow(image,extent=[minx,maxx,maxy,miny])
        if 'masks_individual' in elements:
            plt.imshow(label2rgb(label(mask), bg_label=0),alpha=0.3)
        if not fit_res and do_fit == True:
            print('Fitted axes not found: Attempting axes fit for',fit_method,'...')
            _,_,a_coords,b_coords = measure.fit_grain_axes(props,method=fit_method,padding_size=padding_size)
            fit_res=[_,_,a_coords,b_coords]
        if fit_res:
            _,_,a_coords,b_coords = fit_res[0],fit_res[1],fit_res[2],fit_res[3]
        else:
            a_coords,b_coords = [],[]
        if 'mask' in elements:
            plt.imshow(np.ma.masked_where(mask==0,mask),extent=[minx-padding_size,maxx+padding_size,maxy+padding_size,miny-padding_size],alpha= .5)
        if 'masks_outline' in elements:
            img_pad = measure.image_padding(props[0].image,padding_size=padding_size)
            contours = measure.contour_grain(img_pad)
            for contour in contours:
                    plt.plot(contour[:, 1]-(padding_size-.5)+minx, contour[:, 0]-(padding_size-.5)+miny,'-c',linewidth=1.5)
        if 'convex_hull' in elements:
            convex_image = props[0].convex_image
            conv_pad = measure.image_padding(convex_image,padding_size=padding_size)
                #conv_pad = cv.copyMakeBorder(convex_image.astype(int), padding_size, padding_size, padding_size, padding_size, cv.BORDER_CONSTANT, (0,0,0))
            contours = measure.contour_grain(conv_pad)
            for contour in contours:
                plt.plot(contour[:, 1]-(padding_size-.5)+minx, contour[:, 0]-(padding_size-.5)+miny,'-m',linewidth=1.5)
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
            x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= grains.ell_from_props(props)
            plt.plot(x, y, 'r--', linewidth=1.5)
        if 'ellipse_b' in elements:
            x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= grains.ell_from_props(props)
            plt.plot((x1, x4), (y1, y4), '-b', linewidth=1)
        if 'ellipse_a' in elements:
            x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= grains.ell_from_props(props)
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
    
class uncertainty:
    
    def plot_gsd(df,data_ID='',elem='',cix=0):
        y = np.arange(0.01,1.01,0.01)
        if 'colors' in elem:
            colors = elem['colors']
        else:
            colors = [plt.cm.get_cmap('tab20')(i) for i in range(20)]
        if 'input' in elem:
            plt.plot(df['data'],y, color=colors[cix], label = data_ID)
        if 'median' in elem:
            plt.plot(df['med'],y, color=colors[cix], linestyle = 'dashed',label = 'Percentile median')
        if 'CI_bounds' in elem:
            plt.plot(df['uci'],y, color=colors[cix+1], alpha=.5 ,label = '95% CI')
            plt.plot(df['lci'],y, color=colors[cix+1], alpha=.5 )
        if 'CI_area' in elem:    
            plt.fill_betweenx(y,df['lci'],df['uci'],color=colors[cix+1],alpha=0.5,label='95% CI')
        plt.xlim(0)
        plt.ylim(0.05,1)
