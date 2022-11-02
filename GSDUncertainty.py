import os, time, csv
import numpy as np
from scipy import interpolate as interp
from scipy import stats

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
    """
    def with_bootstrapping(gsd,n=10000,CI_bounds=[2.5,97.5]):
        rand = np.random.choice(gsd, (len(gsd), n))
        gsd_list, med_list, upper_CI, lower_CI = [],[],[],[]
        for p in range(0,100,1):
            perc_dist = np.percentile(rand, p, axis=0)
            perc_dist.sort()
            lower, upper = np.percentile(perc_dist, CI_bounds)
            med = np.percentile(perc_dist, 50)
            inp = np.percentile(gsd,p)
            gsd_list.append(inp), med_list.append(med), upper_CI.append(upper), lower_CI.append(lower)
        return(med_list, upper_CI, lower_CI, gsd_list)

    def MC_with_length_scale(gsd,scale_err,length_err,method='truncnorm',n=1000,cutoff='0',mute=False):
        t = time.time()
        gsd = np.delete(gsd, np.where(gsd <= cutoff))
        res_list = []
                      
        for count in range(0,n,1):
            if count == 0: 
                if mute == False:
                    print('Simulating %s curves'% str(n),'with %s grains each...' %round(len(gsd)))
            # randomize profile
            rand_p = np.random.choice(gsd,len(gsd))
            # fit lognorm distribution to randomized profile                               
            shape,lo_c,s_cale = stats.lognorm.fit(rand_p)      
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
                    if method == 'lognorm':
                        """random length error with stats.lognorm"""
                        # draw b-axis from fitted lognorm
                        rand_gi = stats.lognorm.rvs(s=shape,loc=lo_c,scale=s_cale)
                        # uncertainty funtion
                        rand_length_err = stats.norm.rvs(loc = 0, scale = length_err)
                        rand_g = (rand_gi + rand_length_err) * rand_scale_err
                rand_p_new = np.append(rand_p_new,rand_g)
            res_list.append(rand_p_new)
        elapsed = time.time() - t
        return(count, elapsed, res_list)

    def MC_with_sfm_err_SI(gsd,n,cutoff,method,avg_res,alt,alt_std,point_prec_z,point_prec_std,dom_amp,mute=False):
        t = time.time()
        gsd = np.delete(gsd, np.where(gsd <= cutoff))
        res_list = []
        for count in range(0,n,1):
            if count == 0: 
                if mute == False:
                    print('Simulating %s curves'% str(n),'with %s grains each...' %round(len(gsd)))
            rand_p = np.random.choice(gsd,len(gsd))                                          
            rand_p_new = ([])

            # fit lognorm distribution to randomized profile                               
            shape,lo_c,s_cale = stats.lognorm.fit(rand_p)
            
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
                        """random length error with stats.lognorm"""
                        # fit lognorm to input
                        rand_gi = stats.lognorm.rvs(s=shape,loc=lo_c,scale=s_cale)
                        # uncertainty funtion
                        rand_length_err = stats.norm.rvs(loc = 0, scale = length_std)
                        rand_g = (rand_gi + rand_length_err) * rand_scale_err

                    rand_p_new = np.append(rand_p_new,rand_g)
            """Edge handling"""
            ## Only useful when fitting lognorm distributions (uncomment to use)
            ## filtering values above some maximum threshold
            # rand_p_new = np.delete(rand_p_new,np.where(rand_p_new > 1.5*np.max(gsd)))                                     
            ## filtering values below cutoff threshold
            # rand_p_new = np.delete(rand_p_new,np.where(rand_p_new<cutoff-length_std))
            res_list.append(rand_p_new)
        elapsed = time.time() - t
        if mute == False:
            print('...successfully completed in',np.round(elapsed/60,decimals=1),'minutes.')
        return(res_list)

    def MC_with_sfm_err_OM(gsd,n,cutoff,method,avg_res,om_res,alt,alt_std,point_prec_z,point_prec_std,dom_amp,px_err,px_rms,mute=False):
        t = time.time()
        gsd = np.delete(gsd, np.where(gsd <= cutoff))
        res_list = []
        for count in range(0,n,1):
            if count == 0:
                if mute==False:
                    print('Simulating %s curves'% str(n),'with %s grains each...' %round(len(gsd)))
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
            res_list.append(rand_p_new)
        elapsed = time.time() - t
        if mute == False:
            print('...successfully completed in',np.round(elapsed/60,decimals=1),'minutes.')
        return(res_list)

    def get_MC_percentiles(res_list,gsd,CI_bounds=[2.5,97.5]):                                                   
        # get percentiles in 1% steps (shape = [n[100]])
        gsd=gsd.sort()
        D_list = []
        for k in range(0,len(res_list),1):
            pc = [np.percentile(res_list[k],pi) for pi in range(0,100,1)]
            D_list.append(pc)
        # map list to transposed array (i.e., each line = one D-value, each column = one GSD; shape = 100 x n)
        per_array = np.array(D_list)
        a = per_array.transpose()
        # get estimates for upperCI,lowerCI and median & the percentiles for the input
        gsd_list, med_list, upper_CI, lower_CI = [],[],[],[]
        for j in range(0,len(a),1):
            lower_CI.append(np.percentile(a[j],CI_bounds[0]))
            upper_CI.append(np.percentile(a[j],CI_bounds[1]))
            med_list.append(np.percentile(a[j],50))
            gsd_list.append(np.percentile(gsd,j))                                                               
        return(med_list, upper_CI, lower_CI, gsd_list)

class calculate:

    def gsd_uncertainty(gsd,method=''):
        #do perc_uncert with one of the available methods
        return()     
        
    def dataset_uncertainty(INP_DIR,TAR_DIR='',method='',save_results=True):
        gsd = []
        ID = []
        med_list, upper_CI, lower_CI, gsd_list = random.do_uncert_for_gsd(gsd,method=method)

        if save_results == True:
            try:
                os.makedirs(TAR_DIR)    
            except FileExistsError:
                print(TAR_DIR,  " already exists")
            with open(TAR_DIR + '/' + ID + '_MC_perc_uncertainties.txt', 'w') as f:
                fwriter = csv.writer(f,delimiter=';')
                fwriter.writerow(gsd_list)
                fwriter.writerow(med_list)
                fwriter.writerow(upper_CI)
                fwriter.writerow(lower_CI)
                f.close()
                print('Results for',ID,'successfully saved.')  
