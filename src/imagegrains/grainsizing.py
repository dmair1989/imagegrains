import itertools, os, math
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

from scipy.spatial import distance
from pathlib import Path
from tqdm import tqdm
from skimage import io
from skimage.measure import label, find_contours, regionprops_table, regionprops
from scipy import stats
from natsort import natsorted
from glob import glob

from imagegrains import data_loader

def batch_grainsize(INP_DIR,mask_format='tif',mask_str='',TAR_DIR='',filters=None,mute=False,OT=.5,
properties=['label','area','orientation','minor_axis_length','major_axis_length','centroid','local_centroid'],fit_method='',
return_results=False,save_results=True,do_subfolders=False,do_labels=False,do_predictions=False):
    """ Measures grainsizes in a dataset; can contain subfolders `train`,`test`. If do_subfolders is True, the function will also measure grainsizes in any subfolders of INP_DIR. 

    Parameters
    ----------
    INP_DIR (str) - path to the dataset
    mask_format (str (optional, default ='tif')) - format of the mask images
    mask_str (str (optional, default ='')) - string that is contained in the mask images; e.g., '_mask' for lables, '_pred' for predictions
    TAR_DIR (str (optional, default ='')) - path to the target directory
    filters (dict (optional, default =None)) - dictionary of filters to apply to the grains
    mute (bool (optional, default =False)) - mute the output
    OT (float (optional, default =.5)) - Angular tolerance threshold for b-axis detection during outline fitting in °.
    properties (list (optional, default =['label','area','orientation','minor_axis_length','major_axis_length','centroid','local_centroid'])) - list of properties to be extracted from the masks
    fit_method (str (optional, default ='')) - method to fit the grain outlines. Options are,'convex_hull','mask_outline'. If fit_method is not specified, ellipsoidal fit will be used. ! Please note that using 'convex_hull' or 'mask_outline' will be slow.
    return_results (bool (optional, default =False)) - return the results as a list of pandas dataframes
    save_results (bool (optional, default =True)) - save the results as csv files
    do_subfolders (bool (optional, default =False)) - if True, the function will also measure grainsizes in any subfolders of INP_DIR

    Returns     
    -------
    res_grains_l (list) - list of pandas dataframes containing the results
    res_props_l (list) - list of dictionaries containing the results
    IDs_l (list) - list of IDs

    """
    W_DIR = None
    try:
        dirs = next(os.walk(INP_DIR))[1]
    except StopIteration:
        dirs=[]
        W_DIR = INP_DIR+'/'
    res_grains_l,res_props_l,IDs_l = [],[],[]
    for idx in range(len(dirs)+1):
        if idx < len(dirs):
            if 'train' in dirs[idx]:
                W_DIR = INP_DIR+'/'+str(dirs[idx])
            elif 'test' in dirs[idx]:
                W_DIR = INP_DIR+'/'+str(dirs[idx])
            elif 'pred' in dirs[idx]:
                W_DIR = INP_DIR+'/'+str(dirs[idx])
            elif do_subfolders == True:
                W_DIR = INP_DIR+'/'+str(dirs[idx])
            elif not W_DIR:
                continue
        elif idx == len(dirs) and not res_grains_l:
            W_DIR = INP_DIR+'/'
        if W_DIR:
            res_grains_i,res_props_i,IDs_i= grains_in_dataset(INP_DIR=W_DIR,mask_format=mask_format,mask_str=mask_str,
            TAR_DIR=TAR_DIR,filters=filters,mute=mute,OT=OT,properties=properties,fit_method=fit_method,
            return_results=return_results,save_results=save_results)
            if return_results==True:
                for grains, props, id in zip(res_grains_i,res_props_i,IDs_i):
                    res_grains_l.append(grains)
                    res_props_l.append(props)
                    IDs_l.append(id)
            W_DIR = None
    return res_grains_l,res_props_l,IDs_l

def grains_in_dataset(inp_list=None,INP_DIR=None,mask_format='tif',mask_str='',TAR_DIR='',filters=None,mute=False,OT=.5,
properties=['label','area','orientation','minor_axis_length','major_axis_length','centroid','local_centroid'],fit_method='',
return_results=False,save_results=True,image_res=None,set_id=None):
    """
    Measures grainsizes in a dataset.

    Parameters
    ----------
    INP_DIR (str) - path to the dataset
    mask_format (str (optional, default ='tif')) - format of the mask images
    mask_str (str (optional, default ='')) - string that is contained in the mask images
    TAR_DIR (str (optional, default ='')) - path to the target directory
    filters (dict (optional, default =None)) - dictionary of filters to apply to the grains
    mute (bool (optional, default =False)) - mute the output
    OT (float (optional, default =.5)) - Angular tolerance threshold for b-axis detection during outline fitting in °.
    properties (list (optional, default =['label','area','orientation','minor_axis_length','major_axis_length','centroid','local_centroid'])) - list of properties to be extracted from the masks
    fit_method (str (optional, default ='')) - method to fit the grain outlines. Options are'convex_hull','mask_outline'.
    return_results (bool (optional, default =False)) - return the results as a list of pandas dataframes
    save_results (bool (optional, default =True)) - save the results as csv files
    image_res (list (optional, default =None)) - list of image resolutions in µm/pixel

    
    Returns
    -------
    res_grains (list) - list of pandas dataframes containing the results
    res_props (list) - list of dictionaries containing the results
    IDs (list) - list of IDs

    """
    if not inp_list:
        X = natsorted(glob(INP_DIR+'/*'+mask_str+'*.'+mask_format))
    else:
        X = inp_list
    if not set_id and not inp_list:
        set_id = str(INP_DIR)
    elif set_id:
        set_id = set_id
    else:
        set_id = 'Unknown dataset'
    res_grains,res_props,IDs = [],[],[]
    #for idx,X_i in tqdm(enumerate(X),desc=str(INP_DIR),unit='file',colour='MAGENTA',position=0,leave=True):
    for idx in tqdm(range(len(X)),desc=str(set_id),unit='file',colour='MAGENTA',position=0,leave=True):
        ID = Path(X[idx]).stem
        if 'flow' in ID: #catch flow representation files frpm cp
            if idx==0:
                print('Skipping flow representation files - use mask_str to filter files')
            continue
        else:
            #ID = X_i.split('\\')[len(X_i.split('\\'))-1].split('.')[0]
            masks = label(io.imread(X[idx]))
            if image_res:
                image_res_i=image_res[idx]
            else:
                image_res_i = []
            props_df,props = grains_from_masks(masks,filters=filters,OT=OT,mute=mute,properties=properties,ID=ID,image_res=image_res_i,fit_method=fit_method)
            if save_results == True:
                if TAR_DIR:
                    os.makedirs(TAR_DIR, exist_ok=True)
                    props_df.to_csv(TAR_DIR+'/'+str(ID)+'_grains.csv')
                else:
                    props_df.to_csv(INP_DIR+'/'+str(ID)+'_grains.csv')
            if return_results ==True:
                res_grains.append(props_df),res_props.append(props),IDs.append(ID)
    return res_grains,res_props,IDs 

def grains_from_masks(masks,filters=None,mute=False,OT=.5,fit_method='',image_res=[],ID='',
properties=['label','area','orientation','minor_axis_length','major_axis_length','centroid','local_centroid']):
    """
    Measures grainsizes in single image dataset.
    
    Parameters
    ----------
    masks (numpy array) - numpy array containing the masks
    filters (dict (optional, default =None)) - dictionary of filters to apply to the grains
    mute (bool (optional, default =False)) - mute the output
    OT (float (optional, default =.5)) - Angular tolerance threshold for b-axis detection during outline fitting in °.
    fit_method (str (optional, default ='')) - method to fit the grain outlines. Options are'convex_hull','mask_outline'. 
    image_res (list (optional, default =None)) - list of image resolutions in µm/pixel
    ID (str (optional, default ='')) - ID of the image
    properties (list (optional, default =['label','area','orientation','minor_axis_length','major_axis_length','centroid','local_centroid'])) - list of properties to be extracted from the masks
    
    Returns
    -------
    props_df (pandas dataframe) - pandas dataframe containing the results
    props (list) - list of dictionaries containing the results
    
    """
    masks,num = label(masks,return_num=True)
    if mute==False:
        print(ID,':',str(num),' grains found')
    if filters:
        res_, masks = filter_grains(labels=masks,properties=properties,filters=filters,mask=masks)
        if mute==False:
            print(str(len(res_))+' grains after filtering')
    props = regionprops(label(masks))
    props_df = pd.DataFrame(regionprops_table(masks,properties=properties))
    if any(x in fit_method for x in ['convex_hull','mask_outline']):
        if mute== False:
            print('Fitting axes...')
        a_list,b_list,a_coords,b_coords = fit_grain_axes(props,method=fit_method,OT=OT,mute=mute)
        fit_res = [a_list,b_list,a_coords,b_coords]
        _,props_df = compile_ax_stats(masks,props=props,fit_res=fit_res,properties=properties,mute=mute,ID=ID)
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
    return props_df,props

def compile_ax_stats(grains,props=None,fit_res=None,fit_method='convex_hull',padding_size=2, OT=.5,
        export_results=True, mute = False,
        properties=['label','area','orientation','minor_axis_length','major_axis_length','centroid','local_centroid','bbox'],ID=None):
    """
    Compiles grain statistics from a list of grains.

    Parameters
    ----------
    grains (numpy array) - numpy array containing the grains
    props (list (optional, default =None)) - list of dictionaries containing the results
    fit_res (list (optional, default =None)) - list of results from the grain axes fit
    fit_method (str (optional, default ='convex_hull')) - method to fit the grain outlines. Options are 'ellipse','convex_hull','mask_outline'. Default is 'ellipse'
    padding_size (int (optional, default =2)) - padding size for the grain axes fit
    OT (float (optional, default =.5)) - Angular tolerance threshold for b-axis detection during outline fitting in °.
    export_results (bool (optional, default =True)) - export the results to a pandas dataframe
    mute (bool (optional, default =False)) - mute the output
    properties (list (optional, default =['label','area','orientation','minor_axis_length','major_axis_length','centroid','local_centroid'])) - list of properties to be extracted from the masks

    Returns
    -------
    props_df (pandas dataframe) - pandas dataframe containing the results
    props (list) - list of dictionaries containing the results
    
    """
    if not props:
        props = regionprops(label(grains))
    if export_results == True:
        props_df = pd.DataFrame(regionprops_table(label(grains),properties=properties))
    if not fit_method:
        print('Fitted axes not found: Attempting axes fit for',fit_method,'...')
        a_list,b_list,a_coords,b_coords = fit_grain_axes(props,method=fit_method,padding_size=padding_size,OT=OT,mute=mute,ID=ID)
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
    return props,props_df

def ell_stats(masks,export_results=True,properties=[
        'label','area','orientation','minor_axis_length',
        'major_axis_length','centroid','local_centroid'
        ]):
    """
    Compiles grain statistics from a list of masks.	

    Parameters
    ----------
    masks (list) - list of masks
    export_results (bool (optional, default =True)) - export the results to a pandas dataframe
    properties (list (optional, default =['label','area','orientation','minor_axis_length','major_axis_length','centroid','local_centroid'])) - list of properties to be extracted from the masks

    Returns
    -------
    props_df (pandas dataframe) - pandas dataframe containing the results
    props (list) - list of dictionaries containing the results

    """
    props = regionprops(label(masks))
    if export_results == True:
        props_df = pd.DataFrame(regionprops_table(label(masks),properties=properties))
        return props, props_df
    else:
        return props


def fit_grain_axes(props,method='convex_hull',padding=True,padding_size=2,OT=.5,c_threshold=.5,mute=False,ID=None):
    """
    Fits the grain axes to the grain outlines. 

    Parameters
    ----------
    props (list) - list of dictionaries containing the results
    method (str (optional, default ='convex_hull')) - method to fit the grain outlines. Options are 'ellipse','convex_hull','mask_outline'. Default is 'ellipse'
    padding (bool (optional, default =True)) - pad the grain outlines
    padding_size (int (optional, default =2)) - padding size for each region during the grain axes fit
    OT (float (optional, default =.5)) - Angular tolerance threshold for b-axis detection during outline fitting in °.
    c_threshold (float (optional, default =.5)) - threshold for the b-axis detection during outline fitting
    mute (bool (optional, default =False)) - mute the output


    Returns
    -------
    a_list (list) - list of a-axis lengths
    b_list (list) - list of b-axis lengths
    a_coords (list) - list of local coordinates for the a-axis
    b_coords (list) - list of local coordinates for the b-axis

    Notes
    -----
    Fitting might produce bad results for small grains and for grains with few outline points.

    """
    a_list,a_coords,b_list, b_coords = [],[],[],[]
    counter_l = 0
    ##TODO: loop to parallize for each grain; will need refactoring
    for _idx,props in enumerate(props):
        if method == 'convex_hull':
            mask = props.convex_image
        if method == 'mask_outline':
            mask = props.image
        if padding==True:
            mask = image_padding(mask,padding_size)
        outline = contour_grain(mask,c_threshold)
        a_ax,a_points, b_ax,b_points,counter = get_axes(outline,OT=OT,_idx=_idx,mute=mute,ID=ID)
        a_list.append(a_ax), a_coords.append(a_points)
        b_list.append(b_ax), b_coords.append(b_points)
        counter_l += counter
    result = [a_list,b_list,a_coords,b_coords,counter_l]
    counter_res = result[4]
    if counter_res > 0 and mute == False:
        print(ID,'Number of grains with irregular/small shape:',counter_res)
    return result[0],result[1],result[2],result[3]

def image_padding(grain_img,padding_size=2):
    """
    Pads the image with zeros.
    """
    grain_img_pad = cv.copyMakeBorder(grain_img.astype(int), padding_size,padding_size,padding_size,padding_size, cv.BORDER_CONSTANT, (0,0,0)) 
    return grain_img_pad

def contour_grain(grain_img,c_threshold=.5):
    """
    Finds the contours of the grain image.
    """
    contours = find_contours(grain_img,c_threshold)
    return contours

def get_axes(outline,counter=0,OT=0.05,_idx=0,mute=False,ID=None):
    """"
    Fits the grain axes to the grain outline.
    Takes outline as series of X,Y points.
    """
    a_ax, a_points = get_a_axis(outline)
    a_norm = (a_points[1]-a_points[0]) / a_ax
    b_ax,b_points = iterate_b(outline,a_norm,OT)
    if b_ax == 0:
        counter += 1
    while b_ax == 0:
        OT = OT+.5
        b_ax,b_points = iterate_b(outline,a_norm,OT)
        if b_ax == 0 and OT > 5:
            if mute == False:
                print(ID,'! Irregular grain skipped - check shape of grain @index:',_idx)
            break
    return a_ax, a_points,b_ax,b_points,counter

def get_a_axis(outline):
    """"
    Fits the a-axis to the grain outline.
    Takes outline as series of X,Y points
    """
    dist_matrix = distance.cdist(outline[0], outline[0], 'euclidean') #easy and fast 
    a_ax = np.max(dist_matrix)
    max_keys = np.where(dist_matrix  == np.amax(dist_matrix))
    a_points =[outline[0][max_keys[0][0]],outline[0][max_keys[0][1]]]
    return a_ax,a_points

def iterate_b(outline,a_norm,OT=0.05):
    """
    Fits the b-axis to the grain outline.
    Takes outline as series of X,Y points.

    Parameters
    ----------
    outline (list) - list of X,Y coordinates
    a_norm (list) - normalized a-axis vector
    OT (float (optional, default =.5)) - Angular tolerance threshold for b-axis detection during outline fitting in °.

    Returns
    -------
    b_ax (float) - length of the b-axis
    b_points (list) - list of local coordinates for the b-axis

    Note that this method is slow and will generate some randomness. 
    Consider filtering small grains.
    """
    b_ax = 0
    b_points =[]
    for a, b in itertools.combinations(np.array(outline[0]), 2):
            current_distance = math.dist(a,b)
            # for python < 3.8 use: current_distance = np.linalg.norm(a-b)
            if current_distance  != 0: #catch zero distances
                b_norm = (b-a) / current_distance
                dot_p = np.dot(b_norm,a_norm)
                if dot_p >-1 and dot_p < 1: #catch bad cos values
                    angle = np.degrees(np.arccos(dot_p))
                    if angle > 90-OT and angle < 90+OT:
                        if current_distance > b_ax:
                            b_ax = current_distance
                            b_points = [a,b]
    return b_ax,b_points

def batch_outline(labels,imgs,TAR_DIR='',prop_l=None,filters= None,
                         padding=True,padding_size=2,mute=True,ID='',
                         elements=['binary_mask','image_slice','image_masked','mask_outline']):
    for id_x,(lbl,img) in enumerate(zip(labels,imgs)):
        ID = Path(img).stem
        img = io.imread(img)
        lbl = io.imread(lbl)
        if not prop_l:
           props =None
        export_grain_outline(lbl,img=img,props=None,TAR_DIR=TAR_DIR,filters= filters,
                         padding=padding,padding_size=padding_size,mute=mute,ID=ID,
                         elements=elements)
    

def export_grain_outline(masks,img=None,props=None,method='mask_outline', TAR_DIR='',filters= None,
                         padding=True,padding_size=2,mute=False,ID='',plot_summary=False,
                         elements=['binary_mask','image_slice','image_masked','mask_outline']):
    if not ID:
        if mute == False:
            print('No ID given: Generating random ID...')
        ID = str(np.random.randint(1000,2000))
    TAR_DIR = TAR_DIR+'/'+ID+'/'
    os.makedirs(TAR_DIR, exist_ok=True)
    if not props:
        if mute == False:
            print('No regionprops found: Finding grains...') 
        props = regionprops(label(masks))
    for _idx,props_i in enumerate(props):
        if method == 'convex_hull':
            mask = props_i.convex_image
        if method == 'mask_outline':
            mask = props_i.image
        if padding==True:
            mask = image_padding(mask,padding_size)
            contours = contour_grain(mask)
        else:
            contours = contour_grain(mask)
        miny, minx, maxy, maxx = props_i.bbox
        image_slice= img[props_i.slice].copy()
        if 'binary_mask' in elements:
            io.imsave(TAR_DIR+'/'+ID+'_'+str(props_i.label)+'_mask_outline.tif',mask)
        if 'image_slice' in elements:
            io.imsave(TAR_DIR+'/'+ID+'_'+str(props_i.label)+'_image_slice.png',image_slice)
        if 'image_masked' in elements:
            image_slice[props_i.image==0] = 255
            io.imsave(TAR_DIR+'/'+ID+'_'+str(props_i.label)+'_image_masked.png',image_slice)
        if 'mask_outline' in elements:
            x,y = [],[]
            for contour in contours:
                x.append(contour[:, 1])
                y.append(contour[:, 0])
            x_arr,y_arr= [],[]
            for xi,yi in zip(x[0],y[0]):
                x_arr.append(xi)
                y_arr.append(yi)
            df = pd.DataFrame({'x':x_arr,'y':y_arr})
            df.to_csv(TAR_DIR+'/'+ID+'_'+str(props_i.label)+'_mask_outline.csv',index=False)
        #plt.imshow(image_slice,extent=[minx,maxx,maxy,miny])
        if plot_summary == True:
            plt.imshow(image_slice,extent=[minx,maxx,maxy,miny])
            mask_image = np.ma.masked_where(mask==0,mask)
            plt.imshow(mask_image,extent=[minx,maxx,maxy,miny],alpha= .5) 
            plt.axis('off')
        #notice the different coordinate systems
        # coordinates returned by skimage.find_contours()  always lie at array center (.5!) https://scikit-image.org/docs/0.16.x/api/skimage.measure.html#skimage.measure.find_contours    
            if padding == False:
                for contour in contours:
                    plt.plot(contour[:, 1]-.5+minx, contour[:, 0]-.5+miny,'-c',linewidth=1.5)
            else:
                for contour in contours:
                    plt.plot(contour[:, 1]-(padding_size-.5)+minx, contour[:, 0]-(padding_size-.5)+miny,'-c',linewidth=1.5) 
 

def filter_grains(labels,properties,filters,mask,mute=True):
    """
    Filters grains based on properties and masks.
    Returns a dataframe with filtered grains.

    Parameters
    ----------
    labels (array) - labeled image
    properties (list) - list of properties to be extracted from the labeled image
    filters (dict) - dictionary with filter settings
    mask (array) - mask image
    mute (bool (optional, default = True)) - mute print statements


    Returns
    -------
    filtered (dataframe) - dataframe with filtered grains

    """
    if not labels.any():
        filtered = []
    else:
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
        return filtered,mask

def resample_masks(masks,filters=None,method='wolman',grid_size=None,edge_offset=None,n_rand=100,mute=False):
    """
    Resamples masks to a regular grid.

    Parameters
    ----------
    masks (array) - array of masks
    filters (dict (optional, default = None)) - dictionary with filter settings
    method (str (optional, default = 'wolman')) - method
    grid_size (int (optional, default = None)) - grid size in px
    edge_offset (float (optional, default = None)) - edge offset
    n_rand (int (optional, default = 100)) - number of random samples

    Returns
    -------
    resampled (array) - array of resampled masks
    
    Methods
    -------
    wolman - resamples the masks to a regular grid
    random - resamples the masks to a random grid
    
    """
    w,h = masks.shape[0],masks.shape[1]
    if mute==False:
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
        if mute==False:
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
        if mute==False:
            print('number of resampling points:',len(yy))
        for ii in range(len(yy)):
            a=lbs[(xx[ii]-1).astype(int):xx[ii].astype(int),(yy[ii]-1).astype(int):yy[ii].astype(int)]
            kept_grains.append(a[0][0])
    grains_df = pd.DataFrame(regionprops_table(label(masks)))
    bad_grains = [x for x in grains_df['label'] if x not in kept_grains]
    resampled = masks.copy()
    for lb in range(len(bad_grains)):
        resampled[lbs == bad_grains[lb]]=0
    return resampled,xx,yy

def apply_length_cutoff(gsd,cutoff):
    """
    Applies a length cutoff to the grain size distribution.
    
    Parameters
    ----------
    gsd (array) - grain size distribution
    cutoff (float) - cutoff value in microns

    Returns
    -------
    gsd (array) - filtered grain size distribution

    """
    a = len(gsd)
    gsd = np.delete(gsd,np.where(gsd<cutoff))
    b = len(gsd)
    if a-b !=0:
        print(a-b,' entries dropped')
    return gsd
    
def re_scale_dataset(DIR,resolution= None, camera_parameters= None, gsd_format='csv', gsd_str='grains', return_results=False, save_gsds=True, TAR_DIR=''):
    """
    Rescales a dataset of grain size distributions to a given resolution.
    
    Parameters
    ----------
    DIR (str) - directory containing the grain size distributions
    resolution (list (optional, default = None)) - list of resolutions
    camera_parameters (list (optional, default = None)) - list of camera parameters
    gsd_format (str (optional, default = 'csv')) - format of the grain size distributions
    gsd_str (str (optional, default = 'grains')) - string to identify the grain size distributions
    return_results (bool (optional, default = False)) - returns the rescaled grain size distributions
    save_gsds (bool (optional, default = True)) - saves the rescaled grain size distributions
    TAR_DIR (str (optional, default = '')) - directory to save the rescaled grain size distributions


    Returns
    -------
    rescaled_l (list) - grain size distributions to rescale

    """
    gsds = data_loader.load_grain_set(DIR, gsd_format = gsd_format, gsd_str=gsd_str)
    rescaled_l = []
    for idx,gsd in enumerate(gsds):
        try:
            df = pd.read_csv(gsd,index_col='Unnamed: 0')
        except:
            df = pd.read_csv(gsd)
        if type(resolution) == list:
            resolution_i = resolution[idx]
        else:
            resolution_i = resolution
        if camera_parameters:
            camera_parameters_i=camera_parameters[idx]
        else:
            camera_parameters_i = []
        if not '_re_scaled' in gsd:
            rescaled_df = scale_grains(df,resolution=resolution_i,camera_parameters=camera_parameters_i,GSD_PTH=gsd,return_results=return_results,save_gsds=save_gsds,TAR_DIR=TAR_DIR)
            rescaled_l.append(rescaled_df)
    return rescaled_l

def scale_grains(df,resolution='', ID='', GSD_PTH ='', camera_parameters= {
    'image_distance_m': None, 
    'focal_length_mm': None,
    'sensorH_mm': None,
    'sensorW_mm': None,
    'pixelsW':None,
    'pixelsH':None},return_results=False,save_gsds=True,TAR_DIR=''):
    """
    Rescales a grain size distribution to a given resolution.
    
    Parameters
    ----------
    df (dataframe) - grain size results
    resolution (float (optional, default = '')) - resolution in microns
    ID (str (optional, default = '')) - ID of the grain size distribution
    GSD_DIR (str (optional, default = '')) - full path of the grain size distribution file
    camera_parameters (list (optional, default = {
                        'image_distance_m': None,
                        'focal_length_mm': None,
                        'sensorH_mm': None,
                        'sensorW_mm': None,
                        'pixelsW':None,
                        'pixelsH':None})) - list of camera parameters
    return_results (bool (optional, default = False)) - returns the rescaled grain size distributions
    save_gsds (bool (optional, default = True)) - saves the rescaled grain size distributions
    TAR_DIR (str (optional, default = '')) - directory to save the rescaled grain size distributions
    

    Returns
    -------
    df (dataframe) - rescaled grain size distribution

    """
    if not ID:
        try:
            ID = Path(GSD_PTH).stem
            T_DIR = str(Path(GSD_PTH).parent)
            #ID = GSD_DIR.split('\\')[len(GSD_DIR.split('\\'))-1].split('.')[0]
            #T_DIR = GSD_PTH.split(str(ID))[0]
        except ValueError:
            print('Neither ID nor GSD_PTH were provided')
    if resolution:
        resolution = resolution
    elif camera_parameters['image_distance_m']:
        height_m = camera_parameters['image_distance_m']
        focal_length_mm = camera_parameters['focal_length_mm']
        sensorH_mm = camera_parameters['sensorH_mm']
        sensorW_mm = camera_parameters['sensorW_mm']
        pixelsW = camera_parameters['pixelsW']
        pixelsH = camera_parameters['pixelsH']
        resolution = calculate_camera_res(focal_length_mm, height_m, sensorH_mm, sensorW_mm, pixelsH, pixelsW)
    try: 
        df['ell: a-axis (mm)'] = df['ell: a-axis (px)']*resolution
        df['ell: b-axis (mm)'] = df['ell: b-axis (px)']*resolution
    except KeyError:
        pass    
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
            os.makedirs(TAR_DIR, exist_ok=True)
            df.to_csv(TAR_DIR+'/'+str(ID)+'_re_scaled.csv',sep=';')
        else:
            df.to_csv(T_DIR+'/'+str(ID)+'_re_scaled.csv',sep=';')
    if return_results == False:
        df = []
    return df

def calculate_camera_res(focal_length_mm, height_m, sensorH_mm, sensorW_mm, pixelsH, pixelsW):
    """
    Calculates the resolution of a camera in mm per pixel.
    
    Parameters
    ----------
    focal_length_mm (float) - focal length of the camera in mm
    height_m (float) - height of the camera in m
    sensorH_mm (float) - height of the camera sensor in mm
    sensorW_mm (float) - width of the camera sensor in mm
    pixelsH (int) - number of pixels in the height of the image
    pixelsW (int) - number of pixels in the width of the image
    
    Returns
    -------
    average_res (float) - average resolution of the camera in mm per pixel
    
    """
    fovH_m = (sensorH_mm/focal_length_mm)*height_m
    fovW_m = (sensorW_mm/focal_length_mm)*height_m
    #print("\nFoV: {:0.4f} by {:0.4f} m\n".format(fovH_m, fovW_m))
    X_res, Y_res = np.round(fovH_m*1000/pixelsH, 4), np.round(fovW_m*1000/pixelsW, 4)
    average_res = np.round(np.mean([X_res, Y_res]), 4)
    return average_res


def dataset_object_size(gsds,TAR_DIR='',save_results=True):
    """
    Compiles the object size of all grains in a dataset.
    
    Parameters
    ----------
    gsds (list) - list of grain size distributions
    TAR_DIR (str (optional, default = '')) - directory to save the results
    save_results (bool (optional, default = True)) - save the results
    
    Returns
    -------
    res_df (pandas dataframe) - dataframe containing the object size of all grains in a dataset
    
    """
    ID_l,min_l,max_l,med_l,mean_l = [],[],[],[],[]
    for _,gsd in enumerate(gsds):
        dfi = pd.read_csv(gsd)
        ID_l.append(Path(gsd).stem)
        #ID_l.append(gsd.split('\\')[len(gsd.split('\\'))-1].split('.')[0])
        min_l.append(np.min(dfi['ell: b-axis (px)']))
        max_l.append(np.max(dfi['ell: b-axis (px)']))
        med_l.append(np.median(dfi['ell: b-axis (px)']))
        mean_l.append(np.mean(dfi['ell: b-axis (px)']))
    res_df = pd.DataFrame(list(zip(ID_l,min_l,max_l,med_l,mean_l)),columns=['ID','min','max','med','mean'])
    if save_results ==True:
        os.makedirs(TAR_DIR, exist_ok=True)
        res_df.to_csv(TAR_DIR+'dataset_object_size.csv')
    return res_df

def map_grain_res_to_img(imgs,pred_grains,pred_res_props,pred_IDs,m_string=None,p_string=None):
    new_preds,new_res,new_id = [],[],[]
    """
    Re-orders path lists to of prediction masks to a corresponding image path list.
    This function is needed to correctly match the structure split sets.

    """
    for kk in range(len(imgs)):
        if m_string:
            ID = Path(imgs[kk]).stem.split(m_string)[0]
        else:
            ID = Path(imgs[kk]).stem
        for pred, res, id in zip(pred_grains,pred_res_props,pred_IDs):
                if p_string:
                    ID2 = id.split(p_string)[0]
                else:
                    ID2 = id
                if ID == ID2:
                    new_preds.append(pred)
                    new_res.append(res)
                    new_id.append(id)
    return new_preds,new_res,new_id

def do_gsd(gsd):
    """
    Calculates the GSD for grains in one image.
    """
    if not any(gsd):
        return np.zeros(100)
    else:
        perc_dist = [np.percentile(gsd, p, axis=0) for p in range(100)]
        perc_dist.sort()
    return perc_dist

def gsd_for_set(gsds,column='ell: b-axis (mm)'):
    """
    Calculates the GSD for a set of grains from several images.
    
    Parameters
    ----------
    gsds (list) - list of grain size results
    column (str (optional, default = 'ell: b-axis (mm)')) - column to calculate the GSD from
    
    Returns
    -------
    gsd_l (list) - list of GSDs
    id_l (list) - list of IDs
    """
    gsd_l,id_l = [],[]
    for grains in gsds:
        ID = Path(grains).stem
        raw = pd.read_csv(grains,sep=';')[column]
        gsd = do_gsd(raw)
        gsd_l.append(gsd)
        id_l.append(ID)
    return gsd_l,id_l

# Statistics
def get_key_percs(gsd,perc=[15,50,84,96]):
    return np.round([gsd[perc[i]]for i in range(len(perc))],decimals=1) 

def gsd_test_statistics(gsd1,gsd2,mehtod='ks2amp'):
    if mehtod == 'ks2samp':
        a = stats.ks_2samp(gsd1,gsd2)
        return a.statistic,a.pvalue
    elif mehtod == 'wilcoxon':
        if all(np.array(gsd1)-np.array(gsd2))==0:
            return 0,0
        else:
            a = stats.wilcoxon(gsd1,gsd2)
        return a.statistic,a.pvalue

def get_avg_perc_delta(gsd1,gsd2,metric='mean'):
    if metric == 'mean':
        return np.mean(np.mean((np.array(gsd1)-np.array(gsd2))))
    elif metric == 'median':
        return np.mean(np.median((np.array(gsd1)-np.array(gsd2))))
    
def get_avg_perc_std(gsd1,gsd2,metric='mean'):
    if metric == 'mean':
        return np.mean(np.std((np.array(gsd1)-np.array(gsd2))))
    elif metric == 'median':
        return np.median(np.std((np.array(gsd1)-np.array(gsd2))))
   
def compare_gsds_to_gts(gsds,lbls,units='px',CI=0.05,mute=False,return_std=False):
    counter = 0
    dds,ps,stds = [],[],[]
    if len(gsds) != len(lbls):
        print('Number of gsds and lbls do not match')
        return
    for lbl,gsd in zip(lbls,gsds):
        dd = get_avg_perc_delta(lbl,gsd,metric='mean')
        dds.append(dd)
        a,p = gsd_test_statistics(lbl,gsd,mehtod='ks2samp')
        ps.append(p)
        if return_std == True:
            std = get_avg_perc_std(lbl,gsd,metric='mean')
            stds.append(std)
        counter += 1 if p < CI else 0 
    if mute == False:
        print('For '+str(np.round(((len(lbls)-counter)/(len(lbls))*100),
                                decimals=1))+'% ('+ str(len(lbls)-counter) + '/' 
                                + str(len(lbls)) + ') of tiles grains the size distribution is the same (within a confidence interval of ' + str(1-CI) + ').')
        print('The average percentile difference is',np.round(np.mean(dds),decimals=1),'(+/-',str(np.round(np.std(dds),decimals=1))+')',str(units)+'.')
    if return_std == True:
        return dds,ps,stds
    else:
        return dds,ps

def get_key_CIs(gsd_res,perc=[15,50,84,96]):
    if not any(gsd_res[0]):
        ci_dists = np.zeros(4)
    else:
        ci_dists = np.round([[gsd_res[2][perc[i]],gsd_res[1][perc[i]]] for i in range(len(perc))],decimals=1)
    return ci_dists[0], ci_dists[1], ci_dists[2], ci_dists[3]

def count_ks_hits(df, CI=0.05):
    counter = 0
    for i in range(len(df)):
        counter += 1 if df['ks_p'][i] < CI else 0
    if counter <1:
        return len(df), 100
    else:
        return len(df)-counter, np.round((1-counter/len(df))*100,decimals=1)

def avg_delta(df):
    return np.round(np.mean(df['delta_avg']),decimals=2)

def avg_std(df):
    return np.round(np.mean(df['delta_std']),decimals=2)

def summary_statistics(files,res_dict,id_list,sep=';',unit='mm',axis='b-axis',approximation='ellipse',method='bootstrapping',save_summary=True,data_id='pred'):
    if type(files)==str:
        files = [files]
    summary_df = pd.DataFrame()
    for i,file in enumerate(files):
        grains = data_loader.read_grains(file,sep=sep)
        ID = id_list[i]
        n = len(grains)
        gsd = do_gsd(grains)
        key_p = get_key_percs(gsd)
        key_CI = get_key_CIs(res_dict[ID])
        summary_df = summary_df.append(pd.DataFrame({'Image/Masks':ID,'number of grains':n,
                        'D16':key_p[0],'CI D16 (95%)':str(key_CI[0]),
                        'D50':key_p[1],'CI D50 (95%)':str(key_CI[1]),
                        'D84':key_p[2],'CI D84 (95%)':str(key_CI[2]),
                        'D96':key_p[3],'CI D96 (95%)':str(key_CI[3]),
                        'unit':unit,'axis':axis,'method':method,'grain approximation':approximation},index=[i]))
    if save_summary:
        summary_df.to_csv(str(Path(files[0]).parents[1])+'/'+str(data_id)+'_summary_'+method+'.csv',sep=',',index=False)
    return summary_df