import os, csv
import numpy as np
import pandas as pd
from scipy import interpolate as interp
from scipy import stats
from glob import glob
from natsort import(natsorted)
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

class binom:
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
        Calculates confidence interval under normal
        approximation of Fripp and Diplas (1993)
        (see Eaton et al., 2019 Appendix A - https://doi.org/10.5194/esurf-7-789-2019)
        """
        s = 10 * (np.sqrt((n * p * (1 - p)) / n))
        #z = 1.96
        p_uu = (p*100) + (s * z)
        p_ll = (p*100) - (s * z)
        return(p_ll/100,p_uu/100)

    def pbinom_diff(a, b, n, p):
        """
        cf. Eaton 2019 - Equation (2) - https://doi.org/10.5194/esurf-7-789-2019
        """
        return(stats.binom.cdf(b-1, n, p) - stats.binom.cdf(a-1, n, p))

    def QuantBD(n, p, alpha):
        """
        after Eaton et al. 2019 - https://doi.org/10.5194/esurf-7-789-2019
        https://stats.stackexchange.com/questions/99829/how-to-obtain-a-confidence-interval-for-a-percentile/284970#284970
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
                p_c[i,j] = binom.pbinom_diff(l[i], u[j], n, p)

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
        return(p_l, p_u)

class random:
    """
    -----------------------------------
    Grain Size Distributions: Bootstrapping/Monte Carlo modeling for percentile uncertainty
    -----------------------------------
    following Mair et al. (2022) - https://doi.org/10.5194/esurf-10-953-2022
    """
    def bootstrapping(gsd,num_it=1000,CI_bounds=[2.5,97.5]):
        rand = np.random.choice(gsd, (len(gsd), num_it))
        med_list, upper_CI, lower_CI = [],[],[]
        for p in range(0,100,1):
            perc_dist = np.percentile(rand, p, axis=0)
            perc_dist.sort()
            lower, upper = np.percentile(perc_dist, CI_bounds)
            med = np.percentile(perc_dist, 50)
            med_list.append(med), upper_CI.append(upper), lower_CI.append(lower)
        return(med_list, upper_CI, lower_CI)

    def MC_with_length_scale(gsd,scale_err,length_err,method='truncnorm',num_it=1000,cutoff=0,mute=False):
        if not scale_err or not length_err:
            print('Errors missing!')
        gsd = np.delete(gsd, np.where(gsd <= cutoff))
        res_list = []
        if mute == False:
            print('Simulating %s distributions'% str(num_it),'with %s grains each...' %round(len(gsd)))     
        with Pool(processes=cpu_count()-1) as pool:
            res_list=pool.map(partial(random.MC_loop, gsd=gsd,length_err=length_err,scale_err=scale_err,method=method,cutoff=cutoff), range(num_it))
        return(res_list)
    
    def MC_loop(arg,gsd=[],length_err=0,scale_err=1,method ='truncnorm',cutoff=0):
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
        return(rand_p_new)


    def MC_with_sfm_err_SI(gsd,sfm_error,method='truncnorm',avg_res=1,num_it=1000,cutoff=0,mute=False):
        alt = sfm_error['alt_mean']
        alt_std = sfm_error['alt_std'] 
        point_prec_z = sfm_error['z_err'] 
        point_prec_std = sfm_error['z_err_std']
        dom_amp = sfm_error['dom_amp']
        gsd = np.delete(gsd, np.where(gsd <= cutoff))
        res_list = []
        if mute == False:
            print('Simulating %s distributions'% str(num_it),'with %s grains each...' %round(len(gsd)))
        with Pool(processes=cpu_count()-1) as pool:
            res_list=pool.map(partial(random.MC_SfM_SI_loop,gsd=gsd,avg_res=avg_res,alt=alt,alt_std=alt_std,
            method =method,point_prec_z=point_prec_z,point_prec_std=point_prec_std,dom_amp=dom_amp,cutoff=cutoff), range(num_it))
        return(res_list)

    def MC_SfM_SI_loop(arg,gsd=[],avg_res=1,alt=5,alt_std=1,method ='truncnorm',point_prec_z=1,
        point_prec_std=0.5,dom_amp=0.4,cutoff=0):
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
        return(rand_p_new)

    def MC_with_sfm_err_OM(gsd,sfm_error,method='truncnorm',avg_res=1,num_it=1000,cutoff=0,mute=False):
        alt = sfm_error['alt_mean']
        alt_std = sfm_error['alt_std'] 
        point_prec_z = sfm_error['z_err']
        point_prec_std = sfm_error['z_err_std']
        dom_amp = sfm_error['dom_amp']
        om_res = sfm_error['om_res']
        px_err = sfm_error['pix_err']
        px_rms = sfm_error['pix_rms'] 
        gsd = np.delete(gsd, np.where(gsd <= cutoff))
        res_list = []
        if mute == False:
            print('Simulating %s distributions'% str(num_it),'with %s grains each...' %round(len(gsd)))
        with Pool(processes=cpu_count()-1) as pool:
            res_list=pool.map(partial(random.MC_SfM_OM_loop,gsd=gsd,avg_res=avg_res,alt=alt,alt_std=alt_std,
            method =method,point_prec_z=point_prec_z,point_prec_std=point_prec_std,dom_amp=dom_amp,
            cutoff=cutoff,om_res=om_res,px_err=px_err,px_rms=px_rms), range(num_it))
        return(res_list)

    def MC_SfM_OM_loop(arg,gsd=[],avg_res=1,alt=5,alt_std=1,method ='truncnorm',
        point_prec_z=1,point_prec_std=0.5,dom_amp=0.4,cutoff=0,om_res=1,px_err=1,px_rms=.5):
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
        return(rand_p_new)

    def get_MC_percentiles(res_list,CI_bounds=[2.5,97.5]):                                                   
        # get percentiles in 1% steps (shape = [n[100]])
        D_list = []
        for k in range(0,len(res_list),1):
            if len(res_list[k]) == 0:
                    print('empty GSD skipped')
                    continue
            pc = [np.percentile(res_list[k],pi) for pi in range(0,100,1)]
            D_list.append(pc)
        # map list to transposed array (i.e., each line = one D-value, each column = one GSD; shape = 100 x n)
        per_array = np.array(D_list)
        a = per_array.transpose()
        # get estimates for upperCI,lowerCI and median & the percentiles for the input
        med_list, upper_CI, lower_CI = [],[],[]
        for j in range(0,len(a),1):
            lower_CI.append(np.percentile(a[j],CI_bounds[0]))
            upper_CI.append(np.percentile(a[j],CI_bounds[1]))
            med_list.append(np.percentile(a[j],50))                                                             
        return(med_list, upper_CI, lower_CI)

class calculate:

    def dataset_uncertainty(gsds=[],INP_DIR='',grain_str='_grains',sep=',',column_name='',conv_factor=1,method='bootstrapping',scale_err=[],length_err=[],sfm_error={},num_it=1000,CI_bounds=[2.5,97.5],
    MC_method='truncnorm',MC_cutoff=0,avg_res=[],mute=False,save_results=True,TAR_DIR='',return_results=False,res_dict={},sfm_type=''):
        if INP_DIR:
            gsds = natsorted(glob(INP_DIR+'/*'+grain_str+'*.csv'))
        if not gsds:
            print('No GSD(s) provided!')
            return
        for i in tqdm(range(len(gsds)),desc=str(method),unit='gsd',colour='YELLOW',position=0,leave=True):
            if scale_err:
                if len(scale_err) >1: 
                    scale_err_i=scale_err[i]
                else:
                    scale_err_i=scale_err[0]
            else:
                scale_err_i = []
            if length_err:
                if len(length_err) >1: 
                    length_err_i=length_err[i]
                else:
                    length_err_i = length_err[0]
            else:
                length_err_i = []
            if sfm_error:
                if len(sfm_error)>1:
                    sfm_error_i = sfm_error[i]
                else:
                    sfm_error_i = sfm_error[0]
            else:   
                sfm_error_i = {}
            if avg_res:
                if len(avg_res)>1:
                    avg_res_i = avg_res[i]
                else: 
                    avg_res_i = avg_res[0]
            else:
                avg_res_i = 1
            med_list, upper_CI, lower_CI, gsd_list, ID = calculate.gsd_uncertainty(INP_PATH=gsds[i],sep=sep,column_name=column_name,conv_factor=conv_factor,method=method,scale_err=scale_err_i,length_err=length_err_i,
            sfm_error=sfm_error_i,num_it=num_it,CI_bounds=CI_bounds, MC_method=MC_method,MC_cutoff=MC_cutoff,avg_res=avg_res_i,mute=mute,save_results=save_results,TAR_DIR=TAR_DIR,return_results=True,sfm_type=sfm_type)
            if return_results==True:
                res_dict[ID]=[med_list, upper_CI, lower_CI, gsd_list]
        return(res_dict)

    def gsd_uncertainty(gsd=[],ID='',INP_PATH='',sep=',',column_name='',conv_factor=1,method='bootstrapping',scale_err=[],length_err=[],sfm_error={},num_it=1000,CI_bounds=[2.5,97.5],
    MC_method='truncnorm',MC_cutoff=0,avg_res=1,mute=False,save_results=False,TAR_DIR='',return_results=True,sfm_type=''):
        if len(gsd)==0:
            df = pd.read_csv(INP_PATH , sep=sep)
            if not column_name:
                try:
                    df['ell: b-axis (mm)']
                    column_name = 'ell: b-axis (mm)'
                except:
                    print('No valid column found!')
                    return
            gsd = np.sort(df[column_name].to_numpy())*conv_factor
            ID = INP_PATH.split('\\')[len(INP_PATH.split('\\'))-1].split('.')[0]
        if not avg_res:
            avg_res=1
        med_list, upper_CI, lower_CI, gsd_list = calculate.uncertainty(gsd,method=method,scale_err=scale_err,length_err=length_err,
        sfm_error=sfm_error,num_it=num_it,CI_bounds=CI_bounds,MC_method=MC_method,MC_cutoff=MC_cutoff,avg_res=avg_res,mute=mute,sfm_type=sfm_type)
        if save_results == True:
            if TAR_DIR:
                try:
                    os.makedirs(TAR_DIR)    
                except:
                    pass
                OUT_DIR = TAR_DIR
            elif INP_PATH:
                OUT_DIR = INP_PATH.split('\\')[0]+'/'
            with open(OUT_DIR + '/' + ID + str(method) +'_perc_uncert.txt', 'w') as f:
                fwriter = csv.writer(f,delimiter=';')
                fwriter.writerow(gsd_list)
                fwriter.writerow(med_list)
                fwriter.writerow(upper_CI)
                fwriter.writerow(lower_CI)
                f.close()
                if mute == False:
                    print('Results for',ID+'_perc_uncert','successfully saved.') 
        if return_results == True:
            return(med_list, upper_CI, lower_CI, gsd_list, ID)

    def uncertainty(gsd,method='bootstrapping',scale_err=[],length_err=[],sfm_error={},num_it=1000,CI_bounds=[2.5,97.5],
    MC_method='truncnorm',MC_cutoff=0,avg_res=1,mute=False,sfm_type=''):
        if method == 'bootstrapping':
            med_list, upper_CI, lower_CI = random.bootstrapping(gsd,num_it=num_it,CI_bounds=CI_bounds)
        if method == 'MC':
            res_list = random.MC_with_length_scale(gsd,scale_err,length_err,method=MC_method,num_it=num_it,cutoff=MC_cutoff,mute=mute)
            med_list, upper_CI, lower_CI = random.get_MC_percentiles(res_list,CI_bounds=CI_bounds)
        if method == 'MC_SfM':
            if sfm_type == 'OM':
                res_list = random.MC_with_sfm_err_OM(gsd,sfm_error=sfm_error,avg_res=avg_res,num_it=num_it,cutoff=MC_cutoff,method=MC_method,mute=mute)
                med_list, upper_CI, lower_CI = random.get_MC_percentiles(res_list,CI_bounds=CI_bounds)
            else:
                res_list = random.MC_with_sfm_err_SI(gsd,sfm_error=sfm_error,avg_res=avg_res,num_it=num_it,cutoff=MC_cutoff,method=MC_method,mute=mute)
                med_list, upper_CI, lower_CI = random.get_MC_percentiles(res_list,CI_bounds=CI_bounds)
        #do perc_uncert with one of the available methods
        gsd_list=[]
        for p in range(0,100,1):    
            inp = np.percentile(gsd,p)
            gsd_list.append(inp)
        return(med_list, upper_CI, lower_CI, gsd_list)

    def compile_sfm_error(from_file=''):
        if from_file:
            sfm_err_l,sfm_error_i = [],{}
            err_df = pd.read_csv(from_file)
            for row in range(len(err_df)):    
                for col in err_df:
                    sfm_error_i[col] =err_df[col][row]
                sfm_err_l.append(sfm_error_i)
        return(sfm_err_l)

class load:

    def read_set_unc(PATH,mc_str=''):
        dirs = next(os.walk(PATH))[1]
        G_DIR = []
        if 'test' in dirs:
            G_DIR = [str(PATH+'/test/')]
        if 'train' in dirs:
            G_DIR += [str(PATH+'/train/')]
        if not G_DIR:
            G_DIR = PATH
        mcs,ids=[],[]
        for path in G_DIR:
            mc= natsorted(glob(path+'/*'+mc_str+'*.txt'))
            im= natsorted(glob(path+'/*'+'*.jpg'))
            id_i = [im[i].split('\\')[1].split('.')[0] for i in range(len(im))]
            mcs+=mc
            ids+=id_i
        return(mcs,ids) 

    def read_unc(path,sep=';'):
        df = pd.read_csv(path,sep=sep,)
        df = df.T
        df.reset_index(inplace=True)
        df.columns = ['data','med','uci','lci']
        df = np.round(df.astype('float64'),decimals=1)
        return(df)