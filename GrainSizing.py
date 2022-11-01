
import itertools 
import numpy as np
import pandas as pd
from scipy.spatial import distance
import cv2 as cv
from skimage.measure import label, find_contours, regionprops_table, regionprops



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

    def fit_grain_axes(props,method='convex_hull',padding=True,padding_size=2,OT=.05,c_threshold=.5,mute=False):
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
            a_ax,a_points, b_ax,b_points,counter = measure.get_axes(outline,counter=counter,OT=OT,_idx=_idx)
            a_list.append(a_ax), a_coords.append(a_points)
            b_list.append(b_ax), b_coords.append(b_points)
        if counter > 0 and mute == False:
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
        dist_matrix = distance.cdist(outline[0], outline[0], 'euclidean') #easy and fast 
        a_ax = np.max(dist_matrix)
        max_keys = np.where(dist_matrix  == np.amax(dist_matrix))
        a_points =[outline[0][max_keys[0][0]],outline[0][max_keys[0][1]]]
        return(a_ax,a_points)

    def get_axes(outline,counter=0,OT=0.05,_idx=0):
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

class filter:

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
    
    def resample_masks(masks,filters=[],method='wolman',grid_size=[],edge_offset=[],n_rand=100):
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