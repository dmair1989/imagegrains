import os, csv
import numpy as np
import pandas as pd

from pathlib import Path
from scipy import interpolate as interp
from scipy import stats
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

"""
-----------------------------------
Grain Size Distribution: Binomial & normal approx. for percentile uncertainty
-----------------------------------

partially from:
https://github.com/UP-RS-ESP/PebbleCounts-Application
https://doi.org/10.5194/esurf-7-789-2019
https://stats.stackexchange.com/questions/99829/how-to-obtain-a-confidence-interval-for-a-percentile/284970#284970

"""    
def p_c_fripp(n, p, z):
    """
    Calculates confidence interval under normal approximation of Fripp and Diplas (1993)
    (see Eaton et al., 2019 Appendix A - https://doi.org/10.5194/esurf-7-789-2019)

    Parameters
    ----------
    n (int) - number of samples
    p (float) - percentile
    z (float) - z-score (e.g. 1.96 for 95% CI)

    Returns
    -------
    p_ll (float) - lower confidence bound
    p_uu (float) - upper confidence bound

    """
    s = 10 * (np.sqrt((n * p * (1 - p)) / n))
    #z = 1.96
    p_uu = (p*100) + (s * z)
    p_ll = (p*100) - (s * z)
    return(p_ll/100,p_uu/100)

def pbinom_diff(a, b, n, p):
    """
    cf. Eaton 2019 - Equation (2) - https://doi.org/10.5194/esurf-7-789-2019

    Parameters
    ----------
    a (int) - lower bound
    b (int) - upper bound
    n (int) - number of samples
    p (float) - percentile

    Returns
    -------
    p_c (float) - binomial difference

    """
    return(stats.binom.cdf(b-1, n, p) - stats.binom.cdf(a-1, n, p))

def QuantBD(n, p, alpha):
    """
    Quantile-based binomial difference (cf. Eaton 2019 - Section 2.1.1)
    https://stats.stackexchange.com/questions/99829/how-to-obtain-a-confidence-interval-for-a-percentile/284970#284970

    Parameters
    ----------
    n (int) - number of samples
    p (float) - percentile
    alpha (float) - significance level

    Returns
    -------
    l (int) - lower bound
    u (int) - upper bound
    
    """

    # get the upper and lower confidence bound range
    u = stats.binom.ppf(1 - alpha/2, n, p) + np.array([-2, -1, 0, 1, 2]) + 1
    l = stats.binom.ppf(alpha/2, n, p) + np.array([-2, -1, 0, 1, 2])
    u[u > n] = np.inf
    l[l < 0] = -np.inf

    # get the binomial difference (cf. Eaton 2019 - Equation (2))
    p_c = np.zeros((len(u),len(l)))
    for i in range(len(u)):
        for j in range(len(l)):
            p_c[i,j] = pbinom_diff(l[i], u[j], n, p)

    # handle the edges
    if np.max(p_c) < (1 - alpha):
        i = np.where(p_c == np.max(p_c))
    else:
        i = np.where(p_c == np.min(p_c[p_c >= 1 - alpha]))

    # assymetric bounds (cf. Eaton 2019 - Section 2.1.2)
    # this is the "true" interval with uneven tails
    l = l[i[0]]
    u = u[i[0]]

    # symmetric bounds via interpolation (cf. Eaton 2019 - Section 2.1.3)
    k = np.arange(1, n+1, 1)
    pcum = stats.binom.cdf(k, n, p)
    interp_func = interp.interp1d(pcum, k)
    lu_approx = interp_func([alpha/2, 1 - alpha/2])

    # take the number of measurements and translate
    # to lower and upper percentiles
    p_l, p_u = lu_approx[0]/n, lu_approx[1]/n
    return p_l, p_u


"""
-----------------------------------
Grain Size Distributions: Bootstrapping/Monte Carlo modeling for percentile uncertainty
-----------------------------------
following Mair et al. (2022) - https://doi.org/10.5194/esurf-10-953-2022; please cite if used
"""
def bootstrapping(gsd,num_it=1000,CI_bounds=[2.5,97.5]):
    """
    Bootstrapping for percentile uncertainty
    
    Parameters
    ----------
    gsd (array) - grain size distribution
    num_it (int (optional, default=1000)) - number of iterations
    CI_bounds (list (optional, default=[2.5,97.5])) - confidence interval bounds

    Returns
    -------
    med_list (list) - median of bootstrapped distributions
    upper_CI (list) - upper confidence interval of bootstrapped distributions
    lower_CI (list) - lower confidence interval of bootstrapped distributions
            
    """
    rand = np.random.choice(gsd, (len(gsd), num_it))
    if len(rand) >= 1:
        med_list, upper_CI, lower_CI = [],[],[]
        for p in range(100):
            perc_dist = np.percentile(rand, p, axis=0)
            perc_dist.sort()
            lower, upper = np.percentile(perc_dist, CI_bounds)
            med = np.percentile(perc_dist, 50)
            med_list.append(med), upper_CI.append(upper), lower_CI.append(lower)
    else:
        med_list = np.zeros(100)
        upper_CI = np.zeros(100)
        lower_CI = np.zeros(100)
    return med_list, upper_CI, lower_CI

def MC_with_length_scale(gsd,scale_err,length_err,method='truncnorm',num_it=1000,cutoff=0,mute=False):
    """
    Monte Carlo simulation for percentile uncertainty with length and scale error
    
    Parameters
    ----------
    gsd (array) - grain size distribution
    scale_err (float) - scale error
    length_err (float) - length error
    method (str (optional, default='truncnorm')) - method for random number generation
    num_it (int (optional, default=1000)) - number of iterations
    cutoff (float (optional, default=0)) - cutoff for grain size distribution
    mute (bool (optional, default=False)) - mute print statements

    Returns
    -------
    res_list (list) - list of percentile distributions

    """
    gsd = np.delete(np.array(gsd), np.where(np.array(gsd) <= cutoff))
    res_list = []
    if mute == False:
        print('Simulating %s distributions'% str(num_it),'with %s grains each...' %round(len(gsd)))     
    with Pool(processes=cpu_count()-1) as pool:
        res_list=pool.map_async(partial(MC_loop, gsd=gsd,length_err=length_err,scale_err=scale_err,method=method,cutoff=cutoff), range(num_it)).get(num_it)
    return res_list

def MC_loop(arg,gsd=None,length_err=0,scale_err=.1,method ='truncnorm',cutoff=0):
    """
    Monte Carlo loop for percentile uncertainty with length and scale error
    """
    rand_p = np.random.choice(gsd,len(gsd))
    #shape,lo_c,s_cale = stats.lognorm.fit(rand_p)
    rand_p_new = ([])
    for g in rand_p:
                if length_err==0:
                    rand_gi = g
                else:
                    rand_scale_err = stats.norm.rvs(loc = 1.0, scale = scale_err)
                    if method == 'truncnorm':
                        """random length error with stats.truncnorm"""
                        # truncnorm parameters with cut-off at 'cutoff' and inf 
                        a_g, b_g = (cutoff - g) / length_err, (np.inf - g) / length_err 
                        # uncertainty funtion
                        rand_gi = stats.truncnorm.rvs(a_g, b_g, loc = g, scale = length_err)
                        rand_g = rand_gi * rand_scale_err
    #                if method == 'lognorm':
    #                    """random length error with stats.lognorm"""
    #                    # draw b-axis from fitted lognorm
    #                    rand_gi = stats.lognorm.rvs(s=shape,loc=lo_c,scale=s_cale)
    #                    # uncertainty funtion
    #                    rand_length_err = stats.norm.rvs(loc = 0, scale = length_err)
    #                   rand_g = (rand_gi + rand_length_err) * rand_scale_err
                rand_p_new = np.append(rand_p_new,rand_g)
    return rand_p_new


def MC_with_sfm_err_SI(gsd,sfm_error,method='truncnorm',avg_res=1,num_it=1000,cutoff=0,mute=False):
    """
    Monte Carlo simulation for percentile uncertainty with SfM error

    Parameters
    ----------
    gsd (array) - grain size distribution
    sfm_error (dict) - SfM error dictionary
    method (str (optional, default='truncnorm')) - method for random number generation
    avg_res (float (optional, default=1)) - average resolution
    num_it (int (optional, default=1000)) - number of iterations
    cutoff (float (optional, default=0)) - cutoff for grain size distribution
    mute (bool (optional, default=False)) - mute print statements

    Returns
    -------
    res_list (list) - list of percentile distributions

    """
    alt = sfm_error['alt_mean (m)']
    alt_std = sfm_error['alt_std (m)'] 
    point_prec_z = sfm_error['z_err (m)'] 
    point_prec_std = sfm_error['z_err_std (m)']
    dom_amp = sfm_error['dom_amp (m)']
    gsd = np.delete(gsd, np.where(gsd <= cutoff))
    res_list = []
    if mute == False:
        print('Simulating %s distributions'% str(num_it),'with %s grains each...' %round(len(gsd)))
    with Pool(processes=cpu_count()-1) as pool:
        res_list=pool.map_async(partial(MC_SfM_SI_loop,gsd=gsd,avg_res=avg_res,alt=alt,alt_std=alt_std,
        method =method,point_prec_z=point_prec_z,point_prec_std=point_prec_std,dom_amp=dom_amp,cutoff=cutoff), range(num_it)).get(num_it)
    return res_list

def MC_SfM_SI_loop(arg,gsd=None,avg_res=1,alt=5,alt_std=1,method ='truncnorm',point_prec_z=1,
    point_prec_std=0.5,dom_amp=0.4,cutoff=0):
    """
    Monte Carlo loop for percentile uncertainty with SfM error
    """
    rand_p = np.random.choice(gsd,len(gsd))
    #shape,lo_c,s_cale = stats.lognorm.fit(rand_p)
    rand_p_new = ([])
    rand_p = np.random.choice(gsd,len(gsd))                                      

        # fit lognorm distribution to randomized profile                               
        #shape,lo_c,s_cale = stats.lognorm.fit(rand_p)
    for g in rand_p:
            # taking the diagonal of 2 px as max error; d = std, b-axis_sim = mean
        length_std = np.sqrt(2)*avg_res

        if length_std ==0:
            rand_gi = g
        else:
            #scale_err from SFM uncertainty 
            # altitude prec. estimation for image
            e1 = stats.norm.rvs(loc=0,scale=alt_std)
                # survey-wide point precision from sparse cloud, see also James et al. 2020: https://doi.org/10.1002/esp.4878
            e2 = stats.norm.rvs(loc=0,scale=(point_prec_z + point_prec_std)) 
                # survey-wide systematic 'doming' error amplitude from GCPs see also James et al. 2020: https://doi.org/10.1002/esp.4878
                # amp/4 = std
            e3 = stats.uniform.rvs(loc=0,scale=(dom_amp/4))
                # random scale error
            rand_scale_err = 1 + (e1 + e2 + e3)/alt
                
            if method == 'truncnorm':
                # truncnorm parameters with cut-off at 'cutoff' and inf 
                a_g, b_g = (cutoff - g) / length_std , (np.inf - g) / length_std 
                # uncertainty funtion
                rand_gi = stats.truncnorm.rvs(a_g, b_g, loc = g, scale = length_std)

                # scaling randomizd b-axis_sim with scale_err
                rand_g = rand_gi * rand_scale_err

            if method == 'lognorm':
                print('not implemented yet')
                #"""random length error with stats.lognorm"""
                ## fit lognorm to input
                #rand_gi = stats.lognorm.rvs(s=shape,loc=lo_c,scale=s_cale)
                ## uncertainty funtion
                #rand_length_err = stats.norm.rvs(loc = 0, scale = length_std)
                #rand_g = (rand_gi + rand_length_err) * rand_scale_err

            rand_p_new = np.append(rand_p_new,rand_g)
    return rand_p_new

def MC_with_sfm_err_OM(gsd,sfm_error,method='truncnorm',avg_res=1,num_it=1000,cutoff=0,mute=False):
    """
    Monte Carlo simulation for percentile uncertainty with SfM error
    
    Parameters
    ----------
    gsd (array) - grain size distribution
    sfm_error (dict) - SfM error dictionary
    method (str (optional, default='truncnorm')) - method for random number generation
    avg_res (float (optional, default=1)) - average resolution
    num_it (int (optional, default=1000)) - number of iterations
    cutoff (float (optional, default=0)) - cutoff for grain size distribution
    mute (bool (optional, default=False)) - mute print statements

    Returns
    -------
    res_list (list) - list of percentile distributions

    """
    if not sfm_error:
        print('No SfM error dictionary provided.')
    try:
        alt = sfm_error['alt_mean (m)']
        alt_std = sfm_error['alt_std (m)'] 
        point_prec_z = sfm_error['z_err (m)']
        point_prec_std = sfm_error['z_err_std (m)']
        dom_amp = sfm_error['dom_amp (m)']
        om_res = sfm_error['om_res (mm/px)']
        px_err = sfm_error['pix_err (px)']
        px_rms = sfm_error['pix_rms (px)'] 
        gsd = np.delete(gsd, np.where(gsd <= cutoff))
    except:
        print('SfM error dictionary not complete.')
    res_list = []
    if mute == False:
        print('Simulating %s distributions'% str(num_it),'with %s grains each...' %round(len(gsd)))
    with Pool(processes=cpu_count()-1) as pool:
        res_list=pool.map_async(partial(MC_SfM_OM_loop,gsd=gsd,avg_res=avg_res,alt=alt,alt_std=alt_std,
        method =method,point_prec_z=point_prec_z,point_prec_std=point_prec_std,dom_amp=dom_amp,
        cutoff=cutoff,om_res=om_res,px_err=px_err,px_rms=px_rms), range(num_it)).get(num_it)
    return res_list

def MC_SfM_OM_loop(arg,gsd=None,avg_res=1,alt=5,alt_std=1,method ='truncnorm',
    point_prec_z=1,point_prec_std=0.5,dom_amp=0.4,cutoff=0,om_res=1,px_err=1,px_rms=.5):
    """
    Monte Carlo loop for percentile uncertainty with SfM error
    """
    rand_p = np.random.choice(gsd,len(gsd))                                           
    rand_p_new = ([])
    ## fit lognorm distribution to randomized profile                               
    #shape,loc,s_cale = stats.lognorm.fit(rand_p)
    for g in rand_p:
        # taking the diagonal of 2 px as max error; d = std, b-axis_sim = mean
        length_std = 2*np.sqrt(2)*om_res
        if length_std ==0:
            rand_gi = g
        else:
            #scale_err from SFM uncertainty 
            # altitude prec. estimation for image
            e1 = stats.norm.rvs(loc=0,scale=alt_std)
            # survey-wide point precision from sparse cloud, see also James et al. 2020
            if point_prec_z > alt:
                e2 = stats.norm.rvs(loc=0,scale=(point_prec_z))
            else:
                e2 = stats.norm.rvs(loc=0,scale=point_prec_z + point_prec_std) 
                # survey-wide systematic 'doming' error amplitude from GCPs see also James et al. 2020
            # amp/4 = std
            e3 = stats.uniform.rvs(loc=0,scale=(np.absolute(dom_amp/4)))
            # random scale error
            rand_scale_err = 1 + (e1 + e2 + e3)/alt   
            if method == 'truncnorm':
                # get shape err from pix uncertainty
                a, b = (0 - px_err) / px_rms , (np.inf - px_err) / px_rms
                pix_err = stats.truncnorm.rvs(a, b, loc = px_err, scale = px_rms)
                shape_err =  avg_res*pix_err
                # account for model shape uncertainty with truncnorm
                a_g, b_g = (0 - g) / shape_err , (np.inf - g) / shape_err
                g1 = stats.truncnorm.rvs(a_g, b_g, loc = g, scale = shape_err)
                # scaling with scale_err
                g = g1 * rand_scale_err
                # truncnorm parameters with cut-off at 'cutoff' and inf 
                a_g, b_g = (cutoff - g) / length_std , (np.inf - g) / length_std 
                # uncertainty funtion for supersampling
                rand_gi = stats.truncnorm.rvs(a_g, b_g, loc = g, scale = length_std)
                # missfit error from ellipsoid fit to mask; acceptance criterion >30% misfit;
                # 30% taken as 2 sigma of Gaussian --> scale = 0.3/2
                rand_g = rand_gi #* stats.norm.rvs(loc=1,scale=0.3)
            if method == 'lognorm':
                    print('not implemented yet')
                    #"""random length error with stats.lognorm"""
                    ## fit lognorm to input
                    #rand_gi = stats.lognorm.rvs(s=shape,loc=loc,scale=s_cale)
                    ## uncertainty funtion
                    #rand_length_err = stats.norm.rvs(loc = 0, scale = length_std)
                    ## missfit error from ellipsoid fit to mask; acceptance criterion >30% misfit;
                    #rand_gi = stats.norm.rvs(loc=rand_gi,scale=0.3)
                    #rand_g = (rand_gi + rand_length_err) * rand_scale_err
            rand_p_new = np.append(rand_p_new,rand_g)
        """Edge handling"""
        ## filtering values above some maximum threshold
        #rand_p_new = np.delete(rand_p_new,np.where(rand_p_new > 2*np.max(gsd)))                                     
        ## filtering values below cutoff threshold
        #rand_p_new = np.delete(rand_p_new,np.where(rand_p_new<cutoff-length_std))
    return rand_p_new

def get_MC_percentiles(res_list,CI_bounds=[2.5,97.5],mute=True):
    """
    Get percentiles from MC results
    
    Parameters
    ----------
    res_list (list): list of MC results
    CI_bounds (list (optional, default = [2.5,97.5])): list of upper and lower CI bounds

    Returns
    -------
    med_list (list): list of median values
    upper_CI (list): list of upper CI values
    lower_CI (list): list of lower CI values
            
    """                                                   
    # get percentiles in 1% steps (shape = [n[100]])
    D_list = []
    for _,res in enumerate(res_list):
        if len(res) <1:
            if mute == False:
                print('Empty GSD!')
            continue
        pc = [np.percentile(res,pi) for pi in range(100)]
        D_list.append(pc)
    # map list to transposed array (i.e., each line = one D-value, each column = one GSD; shape = 100 x n)
    per_array = np.array(D_list)
    a = per_array.transpose()
    # get estimates for upperCI,lowerCI and median & the percentiles for the input
    med_list, upper_CI, lower_CI = [],[],[]
    for _,a_i in enumerate(a):
        lower_CI.append(np.percentile(a_i ,CI_bounds[0]))
        upper_CI.append(np.percentile(a_i ,CI_bounds[1]))
        med_list.append(np.percentile(a_i ,50))                                                             
    return med_list, upper_CI, lower_CI


def dataset_uncertainty(gsds=None,inp_dir=None,gsd_id=None,grain_str='_grains',sep=',',column_name='',conv_factor=1,method='bootstrapping',scale_err=0.1,length_err=1,sfm_error=None,num_it=1000,CI_bounds=[2.5,97.5],
MC_method='truncnorm',MC_cutoff=0,avg_res=None,mute=False,save_results=True,tar_dir='',return_results=False,res_dict=None,sfm_type='',id_string=''):
    """
    Calculate uncertainty of a dataset of GSDs

    Parameters
    ----------
    gsds (list (optional, default = None)): list of GSDs
    inp_dir (str (optional, default = '')): path to GSDs
    grain_str (str (optional, default = '_grains')): string to identify GSDs
    sep (str (optional, default = ',')): separator for GSDs
    column_name (str (optional, default = '')): column name of GSDs
    conv_factor (float (optional, default = 1)): conversion factor for GSDs
    method (str (optional, default = 'bootstrapping')): method for uncertainty calculation
    scale_err (list (optional, default = None)): list of scale errors
    length_err (list (optional, default = None)): list of length errors
    sfm_error (list (optional, default = None)): list of sfm errors
    num_it (int (optional, default = 1000)): number of iterations for bootstrapping
    CI_bounds (list (optional, default = [2.5,97.5])): list of upper and lower CI bounds
    MC_method (str (optional, default = 'truncnorm')): method for MC uncertainty calculation
    MC_cutoff (float (optional, default = 0)): cutoff for MC uncertainty calculation
    avg_res (list (optional, default = None)): list of average results
    mute (bool (optional, default = False)): mute output
    save_results (bool (optional, default = True)): save results
    tar_dir (str (optional, default = '')): path to save results
    return_results (bool (optional, default = False)): return results
    res_dict (dict (optional, default = None)): dictionary of results
    sfm_type (str (optional, default = '')): type of sfm

    Returns
    -------
    res_dict (dict): dictionary of results

    """
    if inp_dir:
        #gsds = natsorted(glob(inp_dir+'/*'+grain_str+'*.csv'))
        inp_dir = str(Path(inp_dir).as_posix())
        gsds = natsorted(glob(f'{Path(inp_dir)}/*{grain_str}*.csv'))
    if not gsds:
        print('No GSD(s) provided!')
        return
    if not res_dict:
        res_dict = {}
    for idx in tqdm(range(len(gsds)),desc=f'{column_name} {method}',unit='gsd',colour='BLUE',position=0,leave=True):
        if scale_err:
            if type(scale_err)==list: 
                scale_err_i=scale_err[idx]
            else:
                scale_err_i=scale_err
        if length_err:
            if type(length_err)==list: 
                length_err_i=length_err[idx]
            else:
                length_err_i = length_err
        if sfm_error:
            if len(sfm_error)>1:
                sfm_error_i = sfm_error[idx]
            else:
                sfm_error_i = sfm_error[0]
        else:   
            sfm_error_i = {}
        if avg_res:
            if len(avg_res)>1:
                avg_res_i = avg_res[idx]
            else: 
                avg_res_i = avg_res[0]
        else:
            avg_res_i = 1
        if inp_dir:
            med_list, upper_CI, lower_CI, gsd_list, gsd_id = gsd_uncertainty(inp_path=gsds[idx],sep=sep,column_name=column_name,conv_factor=conv_factor,method=method,scale_err=scale_err_i,length_err=length_err_i,
            sfm_error=sfm_error_i,num_it=num_it,CI_bounds=CI_bounds, MC_method=MC_method,MC_cutoff=MC_cutoff,avg_res=avg_res_i,mute=mute,save_results=save_results,tar_dir=tar_dir,return_results=True,sfm_type=sfm_type,id_string=id_string)
        else:
            if not gsd_id:
                gsd_id_i = str(idx)
            else:
                gsd_id_i = gsd_id[idx]
            if all(np.unique(gsds[idx])) == 0:
                if mute == False:
                    print('Empty GSD')
                if return_results==True:
                    res_dict[str(gsd_id)]=[[], [], [], []]
            else:
                med_list, upper_CI, lower_CI, gsd_list, _ = gsd_uncertainty(gsd=gsds[idx],gsd_id=gsd_id_i,sep=sep,column_name=column_name,conv_factor=conv_factor,method=method,scale_err=scale_err_i,length_err=length_err_i,
                sfm_error=sfm_error_i,num_it=num_it,CI_bounds=CI_bounds, MC_method=MC_method,MC_cutoff=MC_cutoff,avg_res=avg_res_i,mute=mute,save_results=save_results,tar_dir=tar_dir,return_results=True,sfm_type=sfm_type,id_string=id_string)
                if return_results==True:
                    res_dict[str(gsd_id_i)]=[med_list, upper_CI, lower_CI, gsd_list]
    return res_dict

def gsd_uncertainty(gsd=None,gsd_id='',inp_path='',sep=',',column_name='',conv_factor=1,method='bootstrapping',scale_err=0.1,length_err=1,sfm_error=None,num_it=1000,CI_bounds=[2.5,97.5],
MC_method='truncnorm',MC_cutoff=0,avg_res=1,mute=False,save_results=False,tar_dir='',return_results=True,sfm_type='',id_string=''):
    """
    Calculate uncertainty of a GSD. Wrapper for calculate.gsd_uncertainty.

    Parameters
    ----------
    gsd (list (optional, default = None)): list of GSDs
    gsd_id (str (optional, default = '')): ID of GSD
    inp_path (str, Path (optional, default = '')): path to GSD
    sep (str (optional, default = ',')): separator for GSD
    column_name (str (optional, default = '')): column name of GSD
    conv_factor (float (optional, default = 1)): conversion factor for GSD
    method (str (optional, default = 'bootstrapping')): method for uncertainty calculation
    scale_err (list (optional, default = None)): list of scale errors
    length_err (list (optional, default = None)): list of length errors
    sfm_error (list (optional, default = None)): list of sfm errors
    num_it (int (optional, default = 1000)): number of iterations for bootstrapping
    CI_bounds (list (optional, default = [2.5,97.5])): list of upper and lower CI bounds
    MC_method (str (optional, default = 'truncnorm')): method for MC uncertainty calculation
    MC_cutoff (float (optional, default = 0)): cutoff for MC uncertainty calculation
    avg_res (list (optional, default = None)): list of average results
    mute (bool (optional, default = False)): mute output
    save_results (bool (optional, default = True)): save results
    tar_dir (str, Path (optional, default = '')): path to save results
    return_results (bool (optional, default = False)): return results
    sfm_type (str (optional, default = '')): type of sfm

    Returns
    -------
    med_list (list): list of median values
    upper_CI (list): list of upper CI values
    lower_CI (list): list of lower CI values
    gsd_list (list): list of GSDs
    gsd_id (str): ID of GSD

    """
    if all(np.unique(gsd)) == 0:
        if mute == False:
            print('Empty GSD')
        if return_results == True:
            return [], [], [], [], gsd_id
    else:
        if not gsd or type(gsd) == str:
            if not inp_path:
                df = pd.read_csv(gsd , sep=sep)
                gsd_id = Path(gsd).stem
                
            else:
                df = pd.read_csv(inp_path , sep=sep)
                gsd_id = Path(inp_path).stem
            out_dir = os.path.dirname(gsd)
            if not column_name:
                try:
                    df['ell: b-axis (mm)']
                    column_name = 'ell: b-axis (mm)'
                except:
                    pass
                try:
                    df['ell: b-axis (px)']
                    column_name = 'ell: b-axis (px)'
                except:
                    print('No data found!')
                    return
            gsd = np.sort(df[column_name].to_numpy())*conv_factor
            #gsd_id = inp_path.split('\\')[len(inp_path.split('\\'))-1].split('.')[0]
        if not avg_res:
            avg_res=1
        med_list, upper_CI, lower_CI, gsd_list = uncertainty(gsd,method=method,scale_err=scale_err,length_err=length_err,
        sfm_error=sfm_error,num_it=num_it,CI_bounds=CI_bounds,MC_method=MC_method,MC_cutoff=MC_cutoff,avg_res=avg_res,mute=mute,sfm_type=sfm_type)
        if save_results == True:
            if tar_dir != '':
                tar_dir = str(Path(tar_dir).as_posix())
                os.makedirs(Path(tar_dir), exist_ok=True)
                out_dir = tar_dir
            elif tar_dir == '':
                if not inp_path or len(inp_path)==0:
                    out_dir = os.getcwd()
                elif inp_path:
                    out_dir = Path(inp_path).parent
                #out_dir = inp_path.split('\\')[0]+'/'
            outpath = Path(f'{out_dir}/{gsd_id}{method}{id_string}_perc_uncert.txt')
            with open(outpath, 'w') as f:
                fwriter = csv.writer(f,delimiter=';')
                fwriter.writerow(gsd_list)
                fwriter.writerow(med_list)
                fwriter.writerow(upper_CI)
                fwriter.writerow(lower_CI)
                f.close()
                if mute == False:
                    print('Results for',gsd_id + id_string + '_perc_uncert','successfully saved.') 
        if return_results == True:
            return med_list, upper_CI, lower_CI, gsd_list, gsd_id

def uncertainty(gsd,method='bootstrapping',scale_err=0.1,length_err=1,sfm_error=None,num_it=1000,CI_bounds=[2.5,97.5],
MC_method='truncnorm',MC_cutoff=0,avg_res=1,mute=False,sfm_type=''):
    """
    Calculate uncertainty of a GSD. 
    """
    med_list = []
    if method == 'bootstrapping':
        med_list, upper_CI, lower_CI = bootstrapping(gsd,num_it=num_it,CI_bounds=CI_bounds)
    if method == 'MC':
        res_list = MC_with_length_scale(gsd,scale_err,length_err,method=MC_method,num_it=num_it,cutoff=MC_cutoff,mute=mute)
        med_list, upper_CI, lower_CI = get_MC_percentiles(res_list,CI_bounds=CI_bounds, mute = mute)
    if method == 'MC_SfM':
        if sfm_type == 'OM':
            res_list = MC_with_sfm_err_OM(gsd,sfm_error=sfm_error,avg_res=avg_res,num_it=num_it,cutoff=MC_cutoff,method=MC_method,mute=mute)
            med_list, upper_CI, lower_CI = get_MC_percentiles(res_list,CI_bounds=CI_bounds, mute = mute)
        else:
            res_list = MC_with_sfm_err_SI(gsd,sfm_error=sfm_error,avg_res=avg_res,num_it=num_it,cutoff=MC_cutoff,method=MC_method,mute=mute)
            med_list, upper_CI, lower_CI = get_MC_percentiles(res_list,CI_bounds=CI_bounds, mute = mute)
    if len(gsd) >= 1:
        gsd_list = [np.percentile(gsd, p, axis=0) for p in range(100)]
    else:
        gsd_list = np.zeros(100)
    if not med_list:
        med_list = np.zeros(100)
        upper_CI = np.zeros(100)
        lower_CI = np.zeros(100)
    #gsd_list=[]
    #for p in range(100):    
    #    inp = np.percentile(gsd,p)
    #    gsd_list.append(inp)
    return med_list, upper_CI, lower_CI, gsd_list

def compile_sfm_error(from_file=''):
    """
    Compiles sfm error from file.
    """
    if from_file:
        sfm_err_l,sfm_error_i = [],{}
        err_df = pd.read_csv(from_file)
        for row in range(len(err_df)):    
            for col in err_df:
                sfm_error_i[col] =err_df[col][row]
            sfm_err_l.append(sfm_error_i)
    return sfm_err_l

