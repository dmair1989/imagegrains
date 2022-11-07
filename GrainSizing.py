import itertools, os
import numpy as np
import pandas as pd
from scipy.spatial import distance
import cv2 as cv
from tqdm import tqdm
from skimage import io
from skimage.measure import label, find_contours, regionprops_table, regionprops
from natsort import natsorted
from glob import glob


class measure:

    def batch_grainsize(INP_DIR,mask_format='tif',mask_str='',TAR_DIR='',filters={},mute=False,OT=.5,
    properties=['label','area','orientation','minor_axis_length','major_axis_length','centroid','local_centroid'],fit_method='',
    return_results=False,save_results=True,do_subfolders=False):
        dirs = next(os.walk(INP_DIR))[1]
        res_grains_l,res_props_l,IDs_l = [],[],[]
        for dir in dirs:
            if 'train' in dir:
                W_DIR = INP_DIR+'/'+str(dir)
            elif 'test' in dir:
                W_DIR = INP_DIR+'/'+str(dir)
            elif do_subfolders == True:
                W_DIR = INP_DIR+'/'+str(dir)
            else:
                W_DIR = INP_DIR
            res_grains_i,res_props_i,IDs_i= measure.grains_in_dataset(W_DIR,mask_format=mask_format,mask_str=mask_str,
            TAR_DIR=TAR_DIR,filters=filters,mute=mute,OT=OT,properties=properties,fit_method=fit_method,
            return_results=return_results,save_results=save_results)
            if return_results==True:
                for x in range(len(res_grains_i)):
                    res_grains_l.append(res_grains_i[x])
                    res_props_l.append(res_props_i[x])
                    IDs_l.append(IDs_i[x])
        return(res_grains_l,res_props_l,IDs_l)

    def grains_in_dataset(INP_DIR,mask_format='tif',mask_str='',TAR_DIR='',filters={},mute=False,OT=.5,
    properties=['label','area','orientation','minor_axis_length','major_axis_length','centroid','local_centroid'],fit_method='',
    return_results=False,save_results=True,image_res=[]):
        X = natsorted(glob(INP_DIR+'/*'+mask_str+'*.'+mask_format))
        res_grains,res_props,IDs = [],[],[]
        for i in tqdm(range(len(X)),desc=str(INP_DIR),unit='file',colour='MAGENTA',position=0,leave=True):
            ID = X[i].split('\\')[len(X[i].split('\\'))-1].split('.')[0]
            masks = label(io.imread(X[i]))
            if image_res:
                image_res_i=image_res[i]
            else:
                image_res_i = []
            props_df,props = measure.grains_from_masks(masks,filters=filters,OT=OT,mute=mute,properties=properties,ID=ID,image_res=image_res_i,fit_method=fit_method)
            if save_results == True:
                if TAR_DIR:
                    props_df.to_csv(TAR_DIR+'/'+str(ID)+'_grains.csv')
                else:
                    props_df.to_csv(INP_DIR+'/'+str(ID)+'_grains.csv')
            if return_results ==True:
                res_grains.append(props_df),res_props.append(props),IDs.append(ID)
        return(res_grains,res_props,IDs) 

    def grains_from_masks(masks,filters={},mute=False,OT=.5,fit_method='',image_res=[],ID='',
    properties=['label','area','orientation','minor_axis_length','major_axis_length','centroid','local_centroid']):
        masks,num = label(masks,return_num=True)
        if mute==False:
            print(ID,':',str(num),' grains found')
        if filters:
            res_, masks = filter.filter_grains(labels=masks,properties=properties,filters=filters,mask=masks)
            if mute==False:
                print(str(len(res_))+' grains after filtering')
        props = regionprops(label(masks))
        props_df = pd.DataFrame(regionprops_table(masks,properties=properties))
        if any(x in fit_method for x in ['convex_hull','mask_outline']):
            if mute== False:
                print('Fitting axes...')
            a_list,b_list,a_coords,b_coords = measure.fit_grain_axes(props,method=fit_method,OT=OT,mute=mute)
            fit_res = [a_list,b_list,a_coords,b_coords]
            _,props_df = measure.compile_ax_stats(masks,props=props,fit_res=fit_res)
        else:
            props_df = props_df
        if image_res:
            props_df['ell: b-axis (mm)']=props_df['minor_axis_length']*image_res
            props_df['ell: a-axis (mm)']=props_df['major_axis_length']*image_res
            if props_df['mask outline: b axis (px)']:
                props_df['mask outline: b axis (mm)'] = props_df['mask outline: b axis (px)']*image_res
                props_df['mask outline: a axis (mm)'] = props_df['mask outline: a axis (px)']*image_res
            if props_df['convex hull: b axis (px)']:
                props_df['convex hull: b axis (mm)'] = props_df['convex hull: b axis (px)']*image_res
                props_df['convex hull: a axis (mm)'] = props_df['convex hull: a axis (px)']*image_res
        if mute==False:
            print('GSD compiled.')
        #some data cleaning
        try:
            props_df.rename(columns = {
                'minor_axis_length':'ell: b-axis (px)',
                'major_axis_length':'ell: a-axis (px)',
                'centroid-0': 'centerpoint y',
                'centroid-1': 'centerpoint x',
                'local_centroid-0':'local centerpoint y',
                'local_centroid-1':'local centerpoint x',
                'bbox-0': 'bbox y1',
                'bbox-1': 'bbox x1',
                'bbox-2': 'bbox y2',
                'bbox-3': 'bbox x2'
                }, inplace = True)
        except:
            print('Modified DataFrame structure - check results.')
        return(props_df,props)

    def compile_ax_stats(grains,props=[],fit_res=[],fit_method='convex_hull',padding_size=2, OT=.5,
            export_results=True, mute = False,
            properties=['label','area','orientation','minor_axis_length','major_axis_length','centroid','local_centroid','bbox']):
        if not props:
            props = regionprops(label(grains))
        if export_results == True:
            props_df = pd.DataFrame(regionprops_table(label(grains),properties=properties))
        if not fit_method:
            print('Fitted axes not found: Attempting axes fit for',fit_method,'...')
            a_list,b_list,a_coords,b_coords = measure.fit_grain_axes(props,method=fit_method,padding_size=padding_size,OT=OT,mute=mute)
        elif fit_res:
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
        props = measure.regionprops(label(masks))
        if export_results == True:
            props_df = pd.DataFrame(regionprops_table(label(masks),properties=properties))
            return(props, props_df)
        else:
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
    
class scale:

    def re_scale_dataset(DIR,resolution=[],camera_parameters= [],gsd_format='csv',gsd_str='grains',return_results=False,save_gsds=True,TAR_DIR=''):
        gsds = scale.load_grain_set(DIR,gsd_format=gsd_format,gsd_str=gsd_str)
        rescaled_l = []
        for i in range(len(gsds)):
            try:
                df = pd.read_csv(gsds[i],index_col='Unnamed: 0')
            except:
                df = pd.read_csv(gsds[i])
            if len(resolution)> 1:
                resolution_i = resolution[i]
            else:
                resolution_i = resolution[0]
            if camera_parameters:
                camera_parameters_i=camera_parameters[i]
            else:
                camera_parameters_i = []
            rescaled_df = scale.scale_grains(df,resolution=resolution_i,camera_parameters=camera_parameters_i,GSD_DIR=gsds[i],return_results=return_results,save_gsds=save_gsds,TAR_DIR=TAR_DIR)
            rescaled_l.append(rescaled_df)
        return(rescaled_l)

    def load_grain_set(DIR,gsd_format='csv',gsd_str='grains'):
        dirs = next(os.walk(DIR[0]))[1]
        G_DIR = []
        if 'test' in dirs:
                G_DIR = [str(DIR[0]+'/test/')]
        if 'train' in dirs:
                G_DIR += [str(DIR[0]+'/train/')]
        if not G_DIR:
            G_DIR = DIR
        gsds=[]
        for path in G_DIR:
            gsds += scale.gsds_from_folder(path,gsd_format=gsd_format,gsd_str=gsd_str)
        return(gsds)
    
    def gsds_from_folder(PATH,gsd_format='csv',gsd_str='grains'):
        gsds_raw = natsorted(glob(PATH+'/*'+gsd_str+'*.'+gsd_format))
        gsds = []
        for gsd in gsds_raw:
            if 're_scaled' not in gsd:
                gsds.append(gsd)
        if not any(gsds):
            print('Could not load GSDs.')
        return(gsds)

    def scale_grains(df,resolution='', ID='', GSD_DIR ='', camera_parameters= {
        'image_distance_m': [], 
        'focal_length_mm': [],
        'sensorH_mm': [],
        'sensorW_mm': [],
        'pixelsW':[],
        'pixelsH':[]},return_results=False,save_gsds=True,TAR_DIR=''):
        if not ID:
            ID = GSD_DIR.split('\\')[len(GSD_DIR.split('\\'))-1].split('.')[0]
            T_DIR = GSD_DIR.split(str(ID))[0]

        if resolution:
            resolution = resolution
        elif camera_parameters['image_distance_m']:
            height_m = camera_parameters['image_distance_m']
            focal_length_mm = camera_parameters['focal_length_mm']
            sensorH_mm = camera_parameters['sensorH_mm']
            sensorW_mm = camera_parameters['sensorW_mm']
            pixelsW = camera_parameters['pixelsW']
            pixelsH = camera_parameters['pixelsH']
            resolution = scale.calculate_camera_res(focal_length_mm, height_m, sensorH_mm, sensorW_mm, pixelsH, pixelsW)
        try: 
            df['ell: b-axis (mm)']
        except:
            df['ell: a-axis (mm)'] = df['ell: a-axis (px)']*resolution
            df['ell: b-axis (mm)'] = df['ell: b-axis (px)']*resolution
        try:
            df['mask outline: a axis (mm)'] = df['mask outline: a axis (px)']*resolution
            df['mask outline: b axis (mm)'] = df['mask outline: b axis (px)']*resolution
        except KeyError:
            pass
        try:
            df['convex hull: a axis (mm)'] = df['convex hull: a axis (px)']*resolution
            df['convex hull: b axis (mm)'] = df['convex hull: b axis (px)']*resolution
        except KeyError:
            pass
        if save_gsds == True:
            if TAR_DIR:
                df.to_csv(TAR_DIR+'/'+str(ID)+'_grains_re_scaled.csv')
            else:
                df.to_csv(T_DIR+str(ID)+'_grains_re_scaled.csv')
        if return_results == False:
            df = []
        return(df)

    def calculate_camera_res(focal_length_mm, height_m, sensorH_mm, sensorW_mm, pixelsH, pixelsW):
        fovH_m = (sensorH_mm/focal_length_mm)*height_m
        fovW_m = (sensorW_mm/focal_length_mm)*height_m
        #print("\nFoV: {:0.4f} by {:0.4f} m\n".format(fovH_m, fovW_m))
        X_res, Y_res = np.round(fovH_m*1000/pixelsH, 4), np.round(fovW_m*1000/pixelsW, 4)
        average_res = np.round(np.mean([X_res, Y_res]), 4)
        return(average_res)
