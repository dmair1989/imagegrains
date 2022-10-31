
import itertools 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
import cv2 as cv
from skimage.measure import label, find_contours, regionprops_table, regionprops
from skimage.segmentation import mark_boundaries 
from skimage.color import label2rgb
from cellpose import metrics


class measure:
    def compile_ax_stats(grains,props=[],fit_res=[],do_fit=False,fit_method='convex_hull',padding_size=2,
            export_results=True,properties=[
                'label','area','orientation','minor_axis_length',
                'major_axis_length','centroid','local_centroid','bbox'
                ]):
        if not props:
            props = measure.find_grains(grains)
        if export_results == True:
            props_df = pd.DataFrame(regionprops_table(label(grains),properties=properties))
        if not fit_res and do_fit == True:
            print('Fitted axes not found: Attempting axes fit for',fit_method,'...')
            a_list,b_list,a_coords,b_coords = measure.fit_grain_axes(props,method=fit_method,padding_size=padding_size)
        if fit_res:
            a_list,b_list,a_coords,b_coords = fit_res[0],fit_res[1],fit_res[2],fit_res[3]
        else:
            print('error')
        if fit_method == 'mask_outline':
            props_df['mask outline: b axis (px)']=b_list
            props_df['mask outline: a axis (px)']=a_list
            props_df['mask outline: local points for a']=a_coords
            props_df['mask outline: local points for b']=b_coords
        if fit_method == 'convex_hull':
            props_df['convex hull: b axis (px)']=b_list
            props_df['convex hull: a axis (px)']=a_list
            props_df['convex hull: local points for a']=a_coords
            props_df['convex hull: local points for b']=b_coords
        return(props,props_df)

    def ell_stats(masks,export_results=True,properties=[
            'label','area','orientation','minor_axis_length',
            'major_axis_length','centroid','local_centroid'
            ]):
        props = measure.find_grains(masks)
        if export_results == True:
            props_df = pd.DataFrame(regionprops_table(label(masks),properties=properties))
            return(props, props_df)
        else:
            return(props)

    def find_grains(masks):
        props = regionprops(label(masks))
        return(props)

    def fit_grain_axes(props,method='convex_hull',padding=True,padding_size=2,OT=.05,c_threshold=.5,verbose=False):
        """
        Fitting might produce bad results for small grains and for grains with few outline points
        """
        a_list,a_coords,b_list, b_coords = [],[],[],[]
        counter = 0
        for _idx in range(len(props)):
            if method == 'convex_hull':
                mask = props[_idx].convex_image
            if method == 'mask_outline':
                mask = props[_idx].image
            if padding==True:
                mask = measure.image_padding(mask,padding_size)
            outline = measure.contour_grain(mask,c_threshold)
            a_ax,a_points, b_ax,b_points,counter = measure.get_axis(outline,counter=counter,OT=OT,_idx=_idx)
            a_list.append(a_ax), a_coords.append(a_points)
            b_list.append(b_ax), b_coords.append(b_points)
        if counter > 0 and verbose == True:
            print('Number of grains with irregular/small shape:',counter)
        return(a_list,b_list,a_coords,b_coords)        

    def image_padding(grain_img,padding_size=2):
        grain_img_pad = cv.copyMakeBorder(grain_img.astype(int), padding_size,padding_size,padding_size,padding_size, cv.BORDER_CONSTANT, (0,0,0)) 
        return(grain_img_pad)

    def contour_grain(grain_img,c_threshold=.5):
        contours = find_contours(grain_img,c_threshold)
        return(contours)

    def get_a_axis(outline):
        """"
        Takes outline as series of X,Y points
        """
        dist_matrix = distance.cdist(outline[0], outline[0], 'euclidean') #easy and fast way
        a_ax = np.max(dist_matrix)
        max_keys = np.where(dist_matrix  == np.amax(dist_matrix))
        a_points =[outline[0][max_keys[0][0]],outline[0][max_keys[0][1]]]
        return(a_ax,a_points)

    def get_axis(outline,counter=0,OT=0.05,_idx=0):
        """"
        Takes outline as series of X,Y points.
        """
        a_ax, a_points = measure.get_a_axis(outline)
        a_norm = (a_points[1]-a_points[0]) / a_ax
        b_ax,b_points = measure.iterate_b(outline,a_norm,OT)
        if b_ax == 0:
            counter += 1
        while b_ax == 0:
            OT = OT+.5
            b_ax,b_points = measure.iterate_b(outline,a_norm,OT)
            if b_ax == 0 and OT > 5:
                print('! Irregular grain skipped - check shape of grain @index:',_idx)
                break
        return(a_ax, a_points,b_ax,b_points,counter)
    
    def iterate_b(outline,a_norm,OT=0.05):
        """
        Note that this method is slow and will generate some randomness. 
        Consider filtering small grains.
        """
        b_ax = 0
        b_points =[]
        for a, b in itertools.combinations(np.array(outline[0]), 2):
                current_distance = np.linalg.norm(a-b)
                if current_distance  != 0: #catch zero distances
                    b_norm = (b-a) / current_distance
                    dot_p = np.dot(b_norm,a_norm)
                    if dot_p >-1 and dot_p < 1: #catch bad cos values
                        angle = np.degrees(np.arccos(dot_p))
                        if angle > 90-OT and angle < 90+OT:
                            if current_distance > b_ax:
                                b_ax = current_distance
                                b_points = [a,b]
        return(b_ax,b_points)

    def filter_grains(labels,properties,filters,mask,mute=True):
        grains = regionprops_table(labels,properties=properties)
        grains_df = pd.DataFrame(grains)
        if mute==False:
            print(len(grains_df),' grains found')
        if filters['edge'][0] == True:
            """Filter edges based on centroid location"""
            w,l = labels.shape
            edge = filters['edge'][1]
            filtered = grains_df[grains_df['centroid-0']>w*edge]
            filtered = filtered[filtered['centroid-0']<w*(1-edge)]
            filtered = filtered[filtered['centroid-1']>l*edge]
            filtered = filtered[filtered['centroid-1']<l*(1-edge)]
        else:
            filtered = grains_df
        if filters['px_cutoff'][0] == True:
            """Filter by b-axis length """
            filtered = filtered[filtered['minor_axis_length']>filters['px_cutoff'][1]]
        else:
            filtered = filtered
        bad_grains = [x for x in grains_df['label'].values if x not in filtered['label'].values]
        if mute==False:
            print(len(filtered),' grains after filtering')
        for label in bad_grains:
            mask[labels == label]=0
        return(filtered,mask)
    
    def resampling_masks(masks,filters=[],method='wolman',grid_size=[],edge_offset=[],n_rand=100):
        w,h = masks.shape[0],masks.shape[1]
        print('image shape:',h,'x',w)
        lbs = label(masks)
        if filters:
            edge_offset = filters['edge'][1]
        elif edge_offset:
            edge_offset = edge_offset
        else:
            edge_offset = .1
        kept_grains = []
        if method == 'wolman':
            if grid_size:
                grid_size = grid_size
            else:
                grid_size = np.round(h/12)
            x_coords = np.round(np.arange((0+w*edge_offset),(w-w*edge_offset),grid_size))
            y_coords = np.round(np.arange((0+h*edge_offset),(h-h*edge_offset),(grid_size)))
            print('number of Wolman nodes:',len(y_coords ),'x',len(x_coords))        
            xx,yy = np.meshgrid(x_coords,y_coords)
            for ii in range(len(yy)):
                for ki in range(len(xx[0])):
                    x=xx[ii][ki]
                    y=yy[ii][ki]
                    a=lbs[(x-1).astype(int):x.astype(int),(y-1).astype(int):y.astype(int)]
                    kept_grains.append(a[0][0])
        if method == 'random':
            xx = np.random.randint((0+w*edge_offset),(w-w*edge_offset),n_rand)
            yy = np.random.randint((0+h*edge_offset),(h-h*edge_offset),n_rand)
            print('number of resampling points:',len(yy))
            for ii in range(len(yy)):
                a=lbs[(xx[ii]-1).astype(int):xx[ii].astype(int),(yy[ii]-1).astype(int):yy[ii].astype(int)]
                kept_grains.append(a[0][0])
        grains_df = pd.DataFrame(regionprops_table(label(masks)))
        bad_grains = [x for x in grains_df['label'] if x not in kept_grains]
        resampled = masks.copy()
        for lb in range(len(bad_grains)):
            resampled[lbs == bad_grains[lb]]=0
        return(resampled,xx,yy)
    
    def apply_length_cutoff(gsd,cutoff):
        a = len(gsd)
        gsd = np.delete(gsd,np.where(gsd<cutoff))
        b = len(gsd)
        if a-b !=0:
            print(a-b,' entries dropped')
        return(gsd)

class plots:

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
    
    def AP_IoU_plot(eval_results,thresholds=[0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]):    
        res_l= [[] for x in range(len(thresholds))]
        for i  in range(len(eval_results)):
            for j in range(len(thresholds)):
                o = eval_results[i]['ap'][j]
                res_l[j].append(o)
        avg_l,std_ul,std_ll =[],[],[]
        fig = plt.figure(figsize=(3, 5))
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
        plt.xlim(0.5,.9)
        plt.ylim(0,1)
        plt.ylabel('Average precision (AP)')
        plt.xlabel('IoU threshold')
        plt.legend()
        plt.tight_layout()
        return(fig)
    
    def all_grains_plot(masks,elements,props=[], image =[], 
                        fit_res =[],fit_method ='convex_hull',do_fit= False,
                        padding_size=2,figsize=(10,10)):
        fig = plt.figure(figsize=figsize)
        if 'image' in elements and image.any:
            plt.imshow(image)
        if 'masks_individual' in elements:
            plt.imshow(label2rgb(label(masks), bg_label=0),alpha=0.3)
        if not props:
            print('No regionprops found: Finding grains...')
            props = measure.find_grains(masks)
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
                x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= plots.ell_from_props(props,_idx)
                plt.plot(x, y, 'r--', linewidth=1.5)
            if 'ellipse_b' in elements:
                x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= plots.ell_from_props(props,_idx)
                plt.plot((x1, x4), (y1, y4), '-b', linewidth=1)
            if 'ellipse_a' in elements:
                x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= plots.ell_from_props(props,_idx)
                plt.plot((x2, x3), (y2, y3), '-r', linewidth=1)
            if 'ellipse_center' in elements:
                plt.plot(x0, y0, '.g', markersize=2)       
            if 'bbox' in elements:
                bx = (minx, maxx, maxx, minx, minx)
                by = (miny, miny, maxy, maxy, miny)
                plt.plot(bx, by, '-r', linewidth=1,label='bbox')
        plt.axis('off')
        return(fig)

    def single_grain_plot(mask,elements,props=[], image =[], fit_res =[],
                            fit_method ='convex_hull',do_fit= False,
                            padding_size=2,figsize=(8,8)):
        fig = plt.figure(figsize=figsize)
        if not props:
            print('No regionprops found: Finding grains...')
            props = measure.find_grains(mask)
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
            x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= plots.ell_from_props(props)
            plt.plot(x, y, 'r--', linewidth=1.5)
        if 'ellipse_b' in elements:
            x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= plots.ell_from_props(props)
            plt.plot((x1, x4), (y1, y4), '-b', linewidth=1)
        if 'ellipse_a' in elements:
            x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,x,y= plots.ell_from_props(props)
            plt.plot((x2, x3), (y2, y3), '-r', linewidth=1)
        if 'ellipse_center' in elements:
            plt.plot(x0, y0, '.g', markersize=2)       
        if 'bbox' in elements:
            bx = (minx, maxx, maxx, minx, minx)
            by = (miny, miny, maxy, maxy, miny)
            plt.plot(bx, by, '-r', linewidth=2,label='bbox')
        plt.axis('off')
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