#master script for comparing lidar (raw and stats), lasso, & silhs


######### PACKAGES ################

import os
import numpy as np
import scipy.stats as scistats
import matplotlib.pyplot as plt
from netCDF4 import Dataset




######### FUNCTIONS #############

def calculate_variance_epsilon_bu(z,w):

    base_level=3
    num_points=30
    deltaz=np.zeros((num_points))
    deltaz_variance=np.zeros((num_points))
    epsilon=np.zeros((num_points))
    spacer=1
    for k in range(1,num_points):
        deltaz[k]=z[base_level+spacer*k]-z[base_level]
        deltaz_variance[k]=np.var(w[:,base_level+spacer*k]-w[:,base_level],0,ddof=1)
        epsilon[k]=deltaz_variance[k]**(3/2)/(2.**(3/2))/deltaz[k]

    epsilon[0]=np.nan

    return deltaz, deltaz_variance, epsilon

def calculate_variance_epsilon_ud(z,w):

    height_min=200
    height_max=900
    height_init=550

    z_min_array=height_min*np.ones((len(z)))
    z_max_array=height_max*np.ones((len(z)))
    z_init_array=height_init*np.ones((len(z)))
    zmni=np.argmin((z_min_array-z)**2)
    zmxi=np.argmin((z_max_array-z)**2)
    zini=np.argmin((z_init_array-z)**2)

    num_points=min(zini-zmni+1,zmxi-zini+1)

    deltaz=np.zeros((num_points))
    deltaz_variance=np.zeros((num_points))
    deltaz_varsum=np.zeros((num_points))
    deltaz_varmult=np.zeros((num_points))
    deltaz_varmult2=np.zeros((num_points))
    scistatskew=np.zeros((num_points))
    scistatkurt=np.zeros((num_points))
    epsilon=np.zeros((num_points))

    for k in range(1,num_points):
        deltaz[k]=z[zini+k]-z[zini-k]
        deltaz_variance[k]=np.var(w[:,zini+k]-w[:,zini-k],0,ddof=1)
        deltaz_varsum[k]=np.var(w[:,zini+k],0,ddof=1)+np.var(w[:,zini-k],0,ddof=1)
        deltaz_varmult[k]=np.corrcoef(w[:,zini+k],w[:,zini-k])[0,1]
        deltaz_varmult2[k],pvalue=scistats.spearmanr(w[:,zini+k],w[:,zini-k])
        scistatskew[k]=scistats.skew(0.707107*(w[:,zini-k]-w[:,zini+k]))
        scistatkurt[k]=scistats.kurtosis(0.707107*(w[:,zini-k]-w[:,zini+k]))
        epsilon[k]=deltaz_variance[k]**(3/2)/(2.**(3/2))/deltaz[k]

    deltaz_varsum[0]=np.var(w[:,zini],0,ddof=1)+np.var(w[:,zini],0,ddof=1)
    epsilon[0]=np.nan

#    print("deltaz_var = ",deltaz_variance)
#    print("deltaz_varsum = ",deltaz_varsum)
#    print("eps = ",epsilon)
#    print("mean eps",np.nanmean(epsilon))

    return deltaz, deltaz_variance, deltaz_varsum, deltaz_varmult, deltaz_varmult2, epsilon, scistatskew, scistatkurt



######## PATHS TO I/O DATA #############

dl_folder='/glade/work/stepheba/lidar/'
dl_stats=['dl20170627.nc','dl20170717.nc','dl20170728.nc','dl20170923.nc','dl20180911.nc','dl20180917.nc','dl20180918.nc','dl20181002.nc']
dl_raw=['dl20170627_raw.nc','dl20170717_raw.nc','dl20170728_raw.nc','dl20170923_raw.nc','dl20180911_raw.nc','dl20180917_raw.nc','dl20180918_raw.nc','dl20181002_raw.nc']

lasso_folder='/glade/work/stepheba/lasso_orig/'
lasso_les=['2017-06-27','2017-07-17','2017-07-28','2017-09-23/02','2018-09-11/08','2018-09-17/08','2018-09-18/08','2018-10-02/08']

lasso_1m_folder='/glade/scratch/stepheba/lasso_1_min_output/'
lasso_1m_les=['2017-06-27','2017-07-17','2017-07-28','2017-09-23','2018-09-11','2018-09-17','2018-09-18','2018-10-02']

silhs_folder0='/glade/work/stepheba/lasso_silhs/lasso_job_vdc_0/'
silhs_folder0_new='/glade/work/stepheba/lasso_silhs/lasso_job_vdc_0_newcode/'
silhs_folder01='/glade/work/stepheba/lasso_silhs/lasso_job_vdc_01/'
silhs_folder1='/glade/work/stepheba/lasso_silhs/lasso_job_vdc_1/'
silhs_folder10='/glade/work/stepheba/lasso_silhs/lasso_job_vdc_10/'
silhs_folder10_new='/glade/work/stepheba/lasso_silhs/lasso_job_vdc_10_newcode/'
silhs_folder20='/glade/work/stepheba/lasso_silhs/lasso_job_vdc_20/'
silhs_folder30='/glade/work/stepheba/lasso_silhs/lasso_job_vdc_30/'
silhs_folder30_new='/glade/work/stepheba/lasso_silhs/lasso_job_vdc_30_newcode/'
silhs_folder50_new='/glade/work/stepheba/lasso_silhs/lasso_job_vdc_50_newcode/'
silhs_folder100='/glade/work/stepheba/lasso_silhs/lasso_job_vdc_100/'
#active SILHS folder
silhs_folder=silhs_folder10_new
silhs_folder2=silhs_folder50_new  #CHANGE VALUES BELOW!!!!!!!
vdc1=10
vdc2=50
silhs_files=['2017-06-27','2017-07-17','2017-07-28','2017-09-23','2018-09-11','2018-09-17','2018-09-18','2018-10-02']

#check output directory exists
output_path='/glade/work/stepheba/figs'
figs_exists=os.path.exists(output_path)
if not figs_exists:
    os.makedirs(output_path)



############# PROCESSING ###############

# finding the right indices for time and height for all data (i.e. DL,
# DL raw, LASSO, WRF-CLUBB-SILHS)


# -------------------------------------
# ADJUST AS NEEDED --------------------
hour_start=17  #min should be 12 since LASSO and SILHS effectively start at 12
hour_end=22  #max should be 27 since LASSO and SILHS effectively end at 27

height_start=0
height_end=4000
height_end_lidar=1200

# -------------------------------------
# -------------------------------------


second_start=hour_start*3600
second_end=hour_end*3600
minute_start=hour_start*60
minute_end=hour_end*60
lasso_minute_start=(hour_start-12)*60
lasso_minute_end=(hour_end-12)*60
silhs_second_start=(hour_start-12)*3600
silhs_second_end=(hour_end-12)*3600


#defining figures---needs to happen before loop
fig_dl_raw_w=plt.figure(figsize=(14.,7.))
fig_dl_raw_w2=plt.figure(figsize=(14.,7.))
#fig_iso=plt.figure(figsize=(14.,7.))
fig_wp2=plt.figure(figsize=(14.,7.))
fig_wp3=plt.figure(figsize=(14.,7.))
#fig_wskew=plt.figure(figsize=(14.,7.))
#fig_eps=plt.figure(figsize=(14.,7.))
#fig_eps_lasso=plt.figure(figsize=(14.,7.))
#fig_eps_silhs=plt.figure(figsize=(14.,7.))
#fig_d11=plt.figure(figsize=(14.,7.))
fig_d112=plt.figure(figsize=(14.,7.))
#fig_eps_scatter=plt.figure()
#fig_big_wp2=plt.figure(figsize=(48.,14.))
#fig_big_wp3=plt.figure(figsize=(48.,14.))
#fig_big_w=plt.figure(figsize=(26.,14.))
#fig_d11_d22_d33=plt.figure(figsize=(14.,7.))
fig_u_v=plt.figure(figsize=(14.,7.))
fig_chi=plt.figure(figsize=(14.,7.))
fig_w_scatter=plt.figure(figsize=(14.,7.))
#fig_w30_scatter=plt.figure(figsize=(14.,7.))
fig_chi_scatter=plt.figure(figsize=(14.,7.))
fig_w_dl_scatter=plt.figure(figsize=(14.,7.))
fig_w_lasso_scatter=plt.figure(figsize=(14.,7.))
fig_w_hist=plt.figure(figsize=(14.,7.))
fig_chi_hist=plt.figure(figsize=(14.,7.))
fig_mf=plt.figure()
fig_bar=plt.figure()
fig_sciskew=plt.figure(figsize=(14.,7.))
fig_scikurt=plt.figure(figsize=(14.,7.))


#define integral variables (correlation lengths)
integral_dl=np.zeros(8)
integral_lasso=np.zeros(8)
integral_silhs=np.zeros(8)




######### LOOP #############

for case in range(0,8):
    #define the datasets
    dl_raw_data=Dataset(dl_folder+dl_raw[case])
    dl_stats_data=Dataset(dl_folder+dl_stats[case])
    lasso_data=Dataset(lasso_folder+lasso_les[case]+'/raw_model/'+'wrfstat_d01_'+lasso_les[case][0:10]+'_12:00:00.nc')
    lasso_1m_stats_data=Dataset(lasso_1m_folder+lasso_1m_les[case]+'/wrfstat.nc')
    lasso_1m_dflt_data=Dataset(lasso_1m_folder+lasso_1m_les[case]+'/wrfout.nc')
    silhs_data=Dataset(silhs_folder+silhs_files[case]+'/lasso_'+silhs_files[case]+'_nl_lh_sample_points_2D.nc')
    silhs_data2=Dataset(silhs_folder2+silhs_files[case]+'/lasso_'+silhs_files[case]+'_nl_lh_sample_points_2D.nc')
    clubb_zm_data=Dataset(silhs_folder+silhs_files[case]+'/lasso_'+silhs_files[case]+'_zm_wrf.nc')
    clubb_zt_data=Dataset(silhs_folder+silhs_files[case]+'/lasso_'+silhs_files[case]+'_zt_wrf.nc')

    
    #extract time and height variables from all datasets

       #doppler lidar raw
    dl_raw_t=np.array(dl_raw_data.variables['time'])
    dl_raw_z=np.array(dl_raw_data.variables['range'])
    dl_raw_w=np.array(dl_raw_data.variables['radial_velocity'])
    dl_raw_int=np.array(dl_raw_data.variables['intensity'])
    dl_raw_snr=dl_raw_int-1

          #filter the raw lidar data to only keep approx every 60 secs
    for i in range(1,2880):
        next_time=i*60
        if next_time>dl_raw_t[-1]:
            break
        t_idx=np.argmin((next_time*np.ones(len(dl_raw_t))-dl_raw_t)**2)
        if i==1:
            dl_raw_t_filtered=dl_raw_t[t_idx]
            dl_raw_w_filtered=dl_raw_w[t_idx,:]
            dl_raw_snr_filtered=dl_raw_snr[t_idx,:]
        else:
            dl_raw_t_filtered=np.append(dl_raw_t_filtered,dl_raw_t[t_idx])
            dl_raw_w_filtered=np.append(dl_raw_w_filtered,dl_raw_w[t_idx,:],0)
            dl_raw_snr_filtered=np.append(dl_raw_snr_filtered,dl_raw_snr[t_idx,:],0)

          #reshape 2D arrays
    dl_raw_w_filtered=np.reshape(dl_raw_w_filtered,(len(dl_raw_t_filtered),len(dl_raw_z)))
    dl_raw_snr_filtered=np.reshape(dl_raw_snr_filtered,(len(dl_raw_t_filtered),len(dl_raw_z)))

          #redefine raw lidar arrays---prob don't need full raw data at this point
    dl_raw_t=dl_raw_t_filtered
    dl_raw_w=dl_raw_w_filtered
    dl_raw_snr=dl_raw_snr_filtered

       #doppler lidar processed stats
    dl_t=np.array(dl_stats_data.variables['time'])
    dl_z=np.array(dl_stats_data.variables['height'])
    dl_w=np.array(dl_stats_data.variables['w'])
    dl_w2=np.array(dl_stats_data.variables['w_variance'])
    dl_wskew=np.array(dl_stats_data.variables['w_skewness'])
    dl_snr=np.array(dl_stats_data.variables['snr'])
    dl_noise=np.array(dl_stats_data.variables['noise'])
    dl_cbh=np.array(dl_stats_data.variables['dl_cbh'])
    
       #original LASSO
    lasso_t=np.array(lasso_data.variables['XTIME'])
    lasso_z=np.mean(np.array(lasso_data.variables['CSP_Z']),0)
    lasso_z8w=np.mean(np.array(lasso_data.variables['CSP_Z8W']),0)
    lasso_w=np.array(lasso_data.variables['CSV_W'][:,:,125,125])
    lasso_u2=np.array(lasso_data.variables['CSP_U2'])
    lasso_v2=np.array(lasso_data.variables['CSP_V2'])
    lasso_w2=np.array(lasso_data.variables['CSP_W2'])
    lasso_w3=np.array(lasso_data.variables['CSP_W3'])
    lasso_wskew=np.array(lasso_data.variables['CSP_WSKEW'])
    
       #1-minute cheyenne LASSO LES reproduction
    lasso_1m_t=np.array(lasso_1m_dflt_data.variables['XTIME'])
    lasso_1m_z=np.mean(np.array(lasso_1m_stats_data.variables['CSP_Z']),0)
    lasso_1m_z8w=np.mean(np.array(lasso_1m_stats_data.variables['CSP_Z8W']),0)
    lasso_1m_u2=np.array(lasso_1m_stats_data.variables['CSP_U2'])
    lasso_1m_v2=np.array(lasso_1m_stats_data.variables['CSP_V2'])
    lasso_1m_w2=np.array(lasso_1m_stats_data.variables['CSP_W2'])
    lasso_1m_w3=np.array(lasso_1m_stats_data.variables['CSP_W3'])
    lasso_1m_wskew=np.array(lasso_1m_stats_data.variables['CSP_WSKEW'])
    
       #wrf-clubb-silhs lasso coarse-grained runs
    silhs_t=np.array(silhs_data.variables['time'])
    silhs_z=np.array(silhs_data.variables['altitude'])
    silhs_w=np.mean(np.array(silhs_data.variables['w']),(2,3))
    silhs_w30=np.mean(np.array(silhs_data2.variables['w']),(2,3))
    silhs_chi=np.mean(np.array(silhs_data.variables['chi']),(2,3))
    #silhs_chi[silhs_chi<0]=np.nan
    silhs_mf=np.mean(np.array(clubb_zt_data.variables['mixt_frac']),(2,3))
    clubb_wp2=np.mean(np.array(clubb_zm_data.variables['wp2']),(2,3))
    clubb_up2_dp1=np.mean(np.array(clubb_zm_data.variables['up2_dp1']),(2,3))
    clubb_vp2_dp1=np.mean(np.array(clubb_zm_data.variables['vp2_dp1']),(2,3))
    clubb_wp2_dp1=np.mean(np.array(clubb_zm_data.variables['wp2_dp1']),(2,3))
    em_dp1 = -0.5*(clubb_up2_dp1 + clubb_vp2_dp1 + clubb_wp2_dp1)
    clubb_wp3=np.mean(np.array(clubb_zt_data.variables['wp3']),(2,3))
    clubb_wskew=np.mean(np.array(clubb_zt_data.variables['Skw_zt']),(2,3))
    


#####process the time and height data to reflect the desired frame
      #doppler lidar processed stats
    dl_t_start_array=second_start*np.ones((len(dl_t)))
    dl_t_end_array=second_end*np.ones(len(dl_t))
    dl_tsi=np.argmin((dl_t_start_array-dl_t)**2)
    dl_tei=np.argmin((dl_t_end_array-dl_t)**2)
    print('DL processed stats, time:',dl_t[dl_tsi],dl_t[dl_tei])
    
    dl_z_start_array=height_start*np.ones((len(dl_z)))
    dl_z_end_array_special=height_end*np.ones(len(dl_z))
    dl_z_end_array=height_end_lidar*np.ones(len(dl_z))
    dl_zsi=np.argmin((dl_z_start_array-dl_z)**2)
    dl_zei=np.argmin((dl_z_end_array-dl_z)**2)
    dl_zei_special=np.argmin((dl_z_end_array_special-dl_z)**2)
    print('height:',dl_z[dl_zsi],dl_z[dl_zei])
    
       #doppler lidar raw data
    dl_raw_t_start_array=second_start*np.ones((len(dl_raw_t)))
    dl_raw_t_end_array=second_end*np.ones((len(dl_raw_t)))
    dl_raw_tsi=np.argmin((dl_raw_t_start_array-dl_raw_t)**2)
    dl_raw_tei=np.argmin((dl_raw_t_end_array-dl_raw_t)**2)
    print('DL raw, time:',dl_raw_t[dl_raw_tsi],dl_raw_t[dl_raw_tei])
    
    dl_raw_z_start_array=height_start*np.ones((len(dl_raw_z)))
    dl_raw_z_end_array_special=height_end*np.ones(len(dl_raw_z))
    dl_raw_z_end_array=height_end_lidar*np.ones((len(dl_raw_z)))
    dl_raw_zsi=np.argmin((dl_raw_z_start_array-dl_raw_z)**2)
    dl_raw_zei=np.argmin((dl_raw_z_end_array-dl_raw_z)**2)
    dl_raw_zei_special=np.argmin((dl_raw_z_end_array_special-dl_raw_z)**2)
    print('height:',dl_raw_z[dl_raw_zsi],dl_raw_z[dl_raw_zei])
    
       #original LASSO
    lasso_t_start_array=lasso_minute_start*np.ones((len(lasso_t)))
    lasso_t_end_array=lasso_minute_end*np.ones(len(lasso_t))
    lasso_tsi=np.argmin((lasso_t_start_array-lasso_t)**2)
    lasso_tei=np.argmin((lasso_t_end_array-lasso_t)**2)
    print('LASSO orig, time:',lasso_t[lasso_tsi],lasso_t[lasso_tei])
    
    lasso_z_start_array=height_start*np.ones((len(lasso_z)))
    lasso_z_end_array=height_end*np.ones(len(lasso_z))
    lasso_zsi=np.argmin((lasso_z_start_array-lasso_z)**2)
    lasso_zei=np.argmin((lasso_z_end_array-lasso_z)**2)
    print('height:',lasso_z[lasso_zsi],lasso_z[lasso_zei])
    
       #lasso 1-minute output Cheyenne reproduction
    lasso_1m_t_start_array=lasso_minute_start*np.ones((len(lasso_1m_t)))
    lasso_1m_t_end_array=lasso_minute_end*np.ones(len(lasso_1m_t))
    lasso_1m_tsi=np.argmin((lasso_1m_t_start_array-lasso_1m_t)**2)
    lasso_1m_tei=np.argmin((lasso_1m_t_end_array-lasso_1m_t)**2)
    print('LASSO 1m, time:',lasso_1m_t[lasso_1m_tsi],lasso_1m_t[lasso_1m_tei])
    
    lasso_1m_z_start_array=height_start*np.ones((len(lasso_1m_z)))
    lasso_1m_z_end_array=height_end*np.ones(len(lasso_1m_z))
    lasso_1m_zsi=np.argmin((lasso_1m_z_start_array-lasso_1m_z)**2)
    lasso_1m_zei=np.argmin((lasso_1m_z_end_array-lasso_1m_z)**2)
    print('height:',lasso_1m_z[lasso_1m_zsi],lasso_1m_z[lasso_1m_zei])
    
       #wrf-clubb-silhs output
    silhs_t_start_array=silhs_second_start*np.ones((len(silhs_t)))
    silhs_t_end_array=silhs_second_end*np.ones(len(silhs_t))
    silhs_tsi=np.argmin((silhs_t_start_array-silhs_t)**2)
    silhs_tei=np.argmin((silhs_t_end_array-silhs_t)**2)
    print('SILHS orig, time:',silhs_t[silhs_tsi],silhs_t[silhs_tei])
    
    silhs_z_start_array=height_start*np.ones((len(silhs_z)))
    silhs_z_end_array=height_end*np.ones(len(silhs_z))
    silhs_zsi=np.argmin((silhs_z_start_array-silhs_z)**2)
    silhs_zei=np.argmin((silhs_z_end_array-silhs_z)**2)
    print('height:',silhs_z[silhs_zsi],silhs_z[silhs_zei])
 


    #processing my LASSO data to make 30-minute running averages output every ten minutes
    lasso_1m_w=np.array(lasso_1m_stats_data.variables['CSV_W'][lasso_1m_tsi-30:lasso_1m_tei,lasso_1m_zsi:lasso_1m_zei,125,125])
    lasso_1m_w_prime=lasso_1m_w
#    lasso_1m_w_filtered=np.zeros(lasso_1m_tei-lasso_1m_tsi+1,lasso_1m_zei-lasso_1m_zsi+1)
    lasso_1m_w_filtered_mean=np.zeros((30,lasso_1m_zei-lasso_1m_zsi))
    for i in range(0,30):
        lasso_1m_w_filtered_mean[i,:]=np.mean(lasso_1m_w[10*i:10*i+31,:],0)
        lasso_1m_w_prime[10*i:10*i+31,:]=lasso_1m_w[10*i:10*i+31,:]-lasso_1m_w_filtered_mean[i,:]

    lasso_1m_w2=np.zeros((30,lasso_1m_zei-lasso_1m_zsi))
    lasso_1m_w3=np.zeros((30,lasso_1m_zei-lasso_1m_zsi))
    for i in range(0,30):
        lasso_1m_w2[i,:]=np.mean(lasso_1m_w_prime[10*i:10*i+31,:]**2,0)
        lasso_1m_w3[i,:]=np.mean(lasso_1m_w_prime[10*i:10*i+31,:]**3,0)

    
    #load only the part of the 3d vert. velocity that is necessary since it 
    #takes up a lot of space
    lasso_1m_u=np.array(lasso_1m_stats_data.variables['CSV_U'][lasso_1m_tsi:lasso_1m_tei,lasso_1m_zsi:lasso_1m_zei,125,125])
    lasso_1m_v=np.array(lasso_1m_stats_data.variables['CSV_V'][lasso_1m_tsi:lasso_1m_tei,lasso_1m_zsi:lasso_1m_zei,125,125])
    lasso_1m_w=np.array(lasso_1m_stats_data.variables['CSV_W'][lasso_1m_tsi:lasso_1m_tei,lasso_1m_zsi:lasso_1m_zei,125,125])
#    lasso_1m_cf=np.mean(np.array(lasso_1m_dflt_data.variables['CLDFRA'][lasso_1m_tsi:lasso_1m_tei,lasso_1m_zsi:lasso_1m_zei,:,:]),(2,3)) 


    #process lidar stats variables
    dl_w2[abs(dl_w2)>100]=np.nan
    dl_wskew[abs(dl_wskew)>100]=np.nan
    dl_w[dl_noise>1.0]=np.nan
    dl_w[dl_snr<0.008]=np.nan
    dl_w2[dl_noise>1.0]=np.nan
    dl_w2[dl_snr<0.008]=np.nan
    dl_wskew[dl_noise>1.0]=np.nan
    dl_wskew[dl_snr<0.008]=np.nan

    #define wp3 for DL
    dl_w3=dl_wskew*dl_w2**(3/2)

    #process lidar raw variables
    dl_raw_w[dl_raw_snr<0.008]=np.nan


   #process SILHS subcolumns to concatenate the relevant time window across subcolumns
    num_of_columns=silhs_w.shape
    num_of_columns=num_of_columns[2]
    silhs_w_concat=silhs_w[silhs_tsi:silhs_tei,:,0]
    silhs_w_concat30=silhs_w30[silhs_tsi:silhs_tei,:,0]
    silhs_chi_concat=silhs_chi[silhs_tsi:silhs_tei,:,0]
    for i in range(1,num_of_columns):
        silhs_w_concat=np.concatenate((silhs_w_concat,silhs_w[silhs_tsi:silhs_tei,:,i]),0)
        silhs_w_concat30=np.concatenate((silhs_w_concat30,silhs_w30[silhs_tsi:silhs_tei,:,i]),0)
        silhs_chi_concat=np.concatenate((silhs_chi_concat,silhs_chi[silhs_tsi:silhs_tei,:,i]),0)



   #for raw doppler lidar, no need to deal with horizontal or other dimensions
    deltaz_dlraw, deltaz_variance_dlraw, deltaz_varsum_dlraw, deltaz_varmult_dlraw, deltaz_varmult_dlraw2, epsilon_dlraw, scistatskew_dlraw, scistatkurt_dlraw = calculate_variance_epsilon_ud(z=dl_raw_z,w=dl_raw_w[dl_raw_tsi:dl_raw_tei,:])

   #for lasso 1-minute output, need to choose columns within the grid
    deltaz_lasso, deltaz_variance_lasso, deltaz_varsum_lasso, deltaz_varmult_lasso, deltaz_varmult_lasso2, epsilon_lasso, scistatskew_lasso, scistatkurt_lasso = calculate_variance_epsilon_ud(z=lasso_1m_z,w=lasso_1m_w)
    deltaz_lasso, deltaz_variance_lasso_u, deltaz_varsum_lasso_u, deltaz_varmult_lasso_u, deltaz_varmult_lasso_u2, epsilon_lasso_u, scistatskew_lasso_u, scistatkurt_lasso_u = calculate_variance_epsilon_ud(z=lasso_z,w=lasso_1m_u)
    deltaz_lasso, deltaz_variance_lasso_v, deltaz_varsum_lasso_v, deltaz_varmult_lasso_v, deltaz_varmult_lasso_v2, epsilon_lasso_v, scistatskew_lasso_v, scistatkurt_lasso_v = calculate_variance_epsilon_ud(z=lasso_z,w=lasso_1m_v)


   #for silhs output, need to select subcolumns
    deltaz_silhs, deltaz_variance_silhs, deltaz_varsum_silhs, deltaz_varmult_silhs, deltaz_varmult_silhs2, epsilon_silhs, scistatskew_silhs, scistatkurt_silhs = calculate_variance_epsilon_ud(z=silhs_z,w=silhs_w_concat) #w=silhs_w[silhs_tsi:silhs_tei,:,0])

    deltaz_silhs30, deltaz_variance_silhs30, deltaz_varsum_silhs30, deltaz_varmult_silhs30, deltaz_varmult_silhs230, epsilon_silhs30, scistatskew_silhs30, scistatkurt_silhs30 = calculate_variance_epsilon_ud(z=silhs_z,w=silhs_w_concat30) #w=silhs_w30[silhs_tsi:silhs_tei,:,0])


    height_min=200
    height_max=900
    z_min_array=height_min*np.ones((len(silhs_z)))
    z_max_array=height_max*np.ones((len(silhs_z)))
    zmni=np.argmin((z_min_array-silhs_z)**2)
    zmxi=np.argmin((z_max_array-silhs_z)**2)

    em_dp1_mean = np.nanmean(em_dp1[silhs_tsi:silhs_tei,zmni:zmxi],(0,1))


    lasso_u=np.array(lasso_data.variables['CSP_U'])
    lasso_v=np.array(lasso_data.variables['CSP_V'])

    ddelta_z_dlraw=np.zeros(12)
    ddelta_z_lasso=np.zeros(12)
    ddelta_z_silhs=np.zeros(12)
    for i in range(1,12):
        ddelta_z_dlraw[i]=deltaz_dlraw[i]-deltaz_dlraw[i-1]
        ddelta_z_lasso[i]=deltaz_lasso[i]-deltaz_lasso[i-1]
        ddelta_z_silhs[i]=deltaz_silhs[i]-deltaz_silhs[i-1]

    integral_dl[case]=np.sum(ddelta_z_dlraw*deltaz_varmult_dlraw2)
    integral_lasso[case]=np.sum(ddelta_z_lasso*deltaz_varmult_lasso2)
    integral_silhs[case]=np.sum(ddelta_z_silhs*deltaz_varmult_silhs2)

####### FIGURES ###########

    #figure with up2 vp2 wp2 and cloud frac
#    ax_iso=fig_iso.add_subplot(2,4,case+1)
#    ax_iso.plot(np.mean(lasso_u2[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label=r"$\overline{u'^2}")
#    ax_iso.plot(np.mean(lasso_v2[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label=r"$\overline{v'^2}")
#    ax_iso.plot(np.mean(lasso_w2[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label=r"$\overline{w'^2}")
##    ax_iso.plot(np.mean(lasso_1m_cf,0),lasso_1m_z[lasso_1m_zsi:lasso_1m_zei],label='cld_frc')
#    if case > 3:
#        ax_iso.set_xlabel('velocity variance [m2/s2]')
#    if case == 0 or case == 4:
#        ax_iso.set_ylabel('height [m]')
#    ax_iso2 = ax_iso.twiny()  # create a second axis that shares the same x-axis    
#    color = 'tab:red'
#    if case<4:
#        ax_iso2.set_xlabel('cloud frac. [-]', color=color)  # we already handled the x-label with ax1
#    ax_iso2.plot(np.mean(lasso_1m_cf,0),lasso_1m_z[lasso_1m_zsi:lasso_1m_zei],color=color)
#    ax_iso2.tick_params(axis='x', labelcolor=color)
#    ax_iso.set_title(silhs_files[case])
#    if case==3:
#        ax_iso.legend(frameon=False)



    #figure with um and vm
    ax_u_v=fig_u_v.add_subplot(2,4,case+1)
    ax_u_v.plot(np.mean(lasso_u[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label=r'$\overline{u}$')
    ax_u_v.plot(np.mean(lasso_v[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label=r'$\overline{v}$')
    if case > 3:
        ax_u_v.set_xlabel(r'velocity [ms$^{-1}$]')
    if case == 0 or case == 4:
        ax_u_v.set_ylabel('height [m]')
    ax_u_v.set_title(silhs_files[case])
    if case==7:
        ax_u_v.legend(loc='lower right')


    #comparing w's
    w_base=200
    w_height=900
    dl_raw_z_start_array=w_base*np.ones((len(dl_raw_z)))
    dl_raw_z_end_array=w_height*np.ones((len(dl_raw_z)))
    w_dl_raw_zsi=np.argmin((dl_raw_z_start_array-dl_raw_z)**2)
    w_dl_raw_zei=np.argmin((dl_raw_z_end_array-dl_raw_z)**2)
    lasso_1m_z_start_array=w_base*np.ones(len(lasso_1m_z))
    lasso_1m_z_end_array=w_height*np.ones(len(lasso_1m_z))
    w_lasso_1m_zsi=np.argmin((lasso_1m_z_start_array-lasso_1m_z)**2)
    w_lasso_1m_zei=np.argmin((lasso_1m_z_end_array-lasso_1m_z)**2)
    lasso_1m_z_end_array=w_height*np.ones(len(lasso_1m_z[lasso_1m_zsi:lasso_1m_zei]))
    lasso_1m_z_start_array=w_base*np.ones(len(lasso_1m_z[lasso_1m_zsi:lasso_1m_zei]))
    w_3d_lasso_1m_zsi=np.argmin((lasso_1m_z_start_array-lasso_1m_z[lasso_1m_zsi:lasso_1m_zei])**2)
    w_3d_lasso_1m_zei=np.argmin((lasso_1m_z_end_array-lasso_1m_z[lasso_1m_zsi:lasso_1m_zei])**2)
    silhs_z_end_array=w_height*np.ones(len(silhs_z))
    silhs_z_start_array=w_base*np.ones(len(silhs_z))
    w_silhs_zsi=np.argmin((silhs_z_start_array-silhs_z)**2)
    w_silhs_zei=np.argmin((silhs_z_end_array-silhs_z)**2)

    ax_dl_raw_w=fig_dl_raw_w.add_subplot(2,4,case+1)
#    ax_dl_raw_w.plot(np.nanmean(dl_raw_w[dl_raw_tsi:dl_raw_tei,dl_raw_zsi:w_dl_raw_zei],0),dl_raw_z[dl_raw_zsi:w_dl_raw_zei],label='Doppler lidar')
#    ax_dl_raw_w.plot(np.mean(lasso_1m_w[:,0:w_3d_lasso_1m_zei],0),lasso_1m_z[lasso_1m_zsi:w_lasso_1m_zei],label='LASSO')
#    ax_dl_raw_w.plot(np.mean(silhs_w[silhs_tsi:silhs_tei,silhs_zsi:w_silhs_zei,0],0),silhs_z[silhs_zsi:w_silhs_zei],label='SILHS')
    ax_dl_raw_w.plot(dl_raw_w[dl_raw_tsi,w_dl_raw_zsi:w_dl_raw_zei],dl_raw_z[w_dl_raw_zsi:w_dl_raw_zei],'b',label='Lidar')
    ax_dl_raw_w.plot(lasso_1m_w[0,w_3d_lasso_1m_zsi:w_3d_lasso_1m_zei],lasso_1m_z[w_lasso_1m_zsi:w_lasso_1m_zei],'r',label='LES-UWM')
    ax_dl_raw_w.plot(silhs_w[silhs_tsi,w_silhs_zsi:w_silhs_zei,0],silhs_z[w_silhs_zsi:w_silhs_zei],'g',label='SILHS, VDC={}'.format(vdc1))
    ax_dl_raw_w.plot(silhs_w[silhs_tsi,w_silhs_zsi:w_silhs_zei,1:4],silhs_z[w_silhs_zsi:w_silhs_zei],'g')
    if case > 3:
        ax_dl_raw_w.set_xlabel(r'$w$ [ms$^{-1}$]')
    if case == 0 or case == 4:
        ax_dl_raw_w.set_ylabel('height [m]')
    ax_dl_raw_w.set_title(silhs_files[case])
    if case==3:
        ax_dl_raw_w.legend(frameon=False,loc='upper right',prop={'size': 7})


    #extra comparison plot with SILHS vdc=30
    ax_dl_raw_w2=fig_dl_raw_w2.add_subplot(2,4,case+1)
    ax_dl_raw_w2.plot(dl_raw_w[dl_raw_tsi,w_dl_raw_zsi:w_dl_raw_zei],dl_raw_z[w_dl_raw_zsi:w_dl_raw_zei],'b',label='Lidar')
    ax_dl_raw_w2.plot(lasso_1m_w[0,w_3d_lasso_1m_zsi:w_3d_lasso_1m_zei],lasso_1m_z[w_lasso_1m_zsi:w_lasso_1m_zei],'r',label='LES-UWM')
    ax_dl_raw_w2.plot(silhs_w30[silhs_tsi,w_silhs_zsi:w_silhs_zei,0],silhs_z[w_silhs_zsi:w_silhs_zei],'g',label='SILHS, VDC={}'.format(vdc2))
    ax_dl_raw_w2.plot(silhs_w30[silhs_tsi,w_silhs_zsi:w_silhs_zei,1:4],silhs_z[w_silhs_zsi:w_silhs_zei],'g')
    if case > 3:
        ax_dl_raw_w2.set_xlabel(r'$w$ [ms$^{-1}$]')
    if case == 0 or case == 4:
        ax_dl_raw_w2.set_ylabel('height [m]')
    ax_dl_raw_w2.set_title(silhs_files[case])
    if case==3:
        ax_dl_raw_w2.legend(frameon=False,loc='upper right',prop={'size': 7})


    #figure with mixt_frac
    ax_mf=fig_mf.add_subplot()
    ax_mf.plot(np.mean(silhs_mf[silhs_tsi:silhs_tei,w_silhs_zsi:w_silhs_zei],0),silhs_z[w_silhs_zsi:w_silhs_zei],label=silhs_files[case])
    #ax_mf.plot(np.mean(silhs_mf30[silhs_tsi:silhs_tei,w_silhs_zsi:w_silhs_zei],0),silhs_z[w_silhs_zsi:w_silhs_zei],label='VDC=30')
    if case==7:
        ax_mf.set_xlabel('mixture fraction [-]')
        ax_mf.set_ylabel('height [m]')
        ax_mf.legend(frameon=False,loc='upper right')
    #ax_mf.set_title(silhs_files[case])



    # chi noodle plots
    ax_chi=fig_chi.add_subplot(2,4,case+1)
    ax_chi.plot(silhs_chi[silhs_tsi,w_silhs_zsi:w_silhs_zei,0],silhs_z[w_silhs_zsi:w_silhs_zei],'g',label='SILHS')
    ax_chi.plot(silhs_chi[silhs_tsi,w_silhs_zsi:w_silhs_zei,1:8],silhs_z[w_silhs_zsi:w_silhs_zei],'g')
    ax_chi.ticklabel_format(style='sci', axis='x', scilimits=(-2,2))
    if case > 3:
        ax_chi.set_xlabel(r'$\chi$ [kg/kg]')
    if case == 0 or case == 4:
        ax_chi.set_ylabel('height [m]')
    ax_chi.set_title(silhs_files[case])


    #wp2
    ax1=fig_wp2.add_subplot(2,4,case+1)
    ax1.plot(np.nanmean(dl_w2[dl_tsi:dl_tei,dl_zsi:dl_zei],0),dl_z[dl_zsi:dl_zei],label='Lidar VAP')
    ax1.plot(np.mean(lasso_w2[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label='LES')
#    ax1.plot(np.mean(lasso_1m_w2[lasso_1m_tsi:lasso_1m_tei,lasso_1m_zsi:lasso_1m_zei],0),lasso_z[lasso_1m_zsi:lasso_1m_zei],label='LASSO 1m')
    ax1.plot(np.mean(clubb_wp2[silhs_tsi:silhs_tei,silhs_zsi:silhs_zei],0),silhs_z[silhs_zsi:silhs_zei],label='CLUBB')
    #plt.imshow(np.mean(lasso_w,(2,3)).T,origin='lower')
    if case > 3:
        ax1.set_xlabel(r"$\overline{w'^2}$ [m$^2$s$^{-2}$]")
    if case == 0 or case == 4:
        ax1.set_ylabel('height [m]')
    ax1.set_title(silhs_files[case])
    if case==3:
        ax1.legend(frameon=False)
    #plt.subplots_adjust(bottom=0.2)


    #wp3
    ax3=fig_wp3.add_subplot(2,4,case+1)
    ax3.plot(np.nanmean(dl_w3[dl_tsi:dl_tei,dl_zsi:dl_zei],0),dl_z[dl_zsi:dl_zei],label='Lidar VAP')
    ax3.plot(np.mean(lasso_w3[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label='LES')
#    ax3.plot(np.mean(lasso_1m_w3[lasso_1m_tsi:lasso_1m_tei,lasso_1m_zsi:lasso_1m_zei],0),lasso_z[lasso_1m_zsi:lasso_1m_zei],label='LASSO 1m')
    ax3.plot(np.mean(clubb_wp3[silhs_tsi:silhs_tei,silhs_zsi:silhs_zei],0),silhs_z[silhs_zsi:silhs_zei],label='CLUBB')
    #plt.imshow(np.mean(lasso_w,(2,3)).T,origin='lower')
    if case > 3:
        ax3.set_xlabel(r"$\overline{w'^3}$ [m$^3$s$^{-3}$]")
    if case == 0 or case == 4:
        ax3.set_ylabel('height [m]')
    ax3.set_title(silhs_files[case])
    if case==3:
        ax3.legend(frameon=False)
    #plt.subplots_adjust(bottom=0.2)


    #skewness
#    ax2=fig_wskew.add_subplot(2,4,case+1)
#    ax2.plot(np.nanmean(dl_wskew[dl_tsi:dl_tei,dl_zsi:dl_zei],0),dl_z[dl_zsi:dl_zei],label='Lidar VAP')
#    ax2.plot(np.mean(lasso_wskew[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label='LES')
##    ax2.plot(np.mean(lasso_1m_wskew[lasso_1m_tsi:lasso_1m_tei,lasso_1m_zsi:lasso_1m_zei],0),lasso_z[lasso_1m_zsi:lasso_1m_zei],label='LASSO 1m')
#    ax2.plot(np.mean(clubb_wskew[silhs_tsi:silhs_tei,silhs_zsi:silhs_zei],0),silhs_z[silhs_zsi:silhs_zei],label='CLUBB')
#    #plt.imshow(np.mean(lasso_w,(2,3)).T,origin='lower')
#    if case > 3:
#        ax2.set_xlabel('w_skewness [-]')
#    if case == 0 or case == 4:
#        ax2.set_ylabel('height [m]')
#    ax2.set_title(silhs_files[case])
#    if case==3:
#        ax2.legend(frameon=False)
#    #plt.subplots_adjust(bottom=0.2)
#
#
#
#    factor23=0.005
#    #figure showing D_11 and epsilon DL raw
##    ax_eps=fig_eps.add_subplot(2,4,case+1)
##    color = 'tab:red'
##    if case>3:
##        ax_eps.set_xlabel(r'$\Delta z$ [m]')
##    if case==0 or case==4:
##        ax_eps.set_ylabel(r'$D_{11}$ [m2/s2]', color=color)
##    last=len(deltaz_dlraw)
##    ax_eps.plot(deltaz_dlraw[1:last],deltaz_variance_dlraw[1:last],color=color)
##    ax_eps.plot(deltaz_dlraw[1:last],deltaz_varsum_dlraw[1:last],color=color,linestyle='dashed')
##    ax_eps.plot(deltaz_dlraw[1:last],factor23*deltaz_dlraw[1:last]**(2/3),color='k',linestyle='dotted')
##    ax_eps.tick_params(axis='y', labelcolor=color)
##
##    ax_eps2 = ax_eps.twinx()
##    color = 'tab:blue'
##    if case==3 or case==7:
##        ax_eps2.set_ylabel(r'$\epsilon$ [m2/s3]', color=color)  
##    ax_eps2.plot(deltaz_dlraw[1:last],epsilon_dlraw[1:last],color=color)
##    ax_eps2.tick_params(axis='y', labelcolor=color)
##    ax_eps2.ticklabel_format(scilimits=[-3,3])
##    ax_eps.set_title(silhs_files[case])
#
#
#    #figure showing D_11 and epsilon LASSO 1-m
##    ax_eps_lasso=fig_eps_lasso.add_subplot(2,4,case+1)
##    color = 'tab:red'
##    if case>3:
##        ax_eps_lasso.set_xlabel(r'$\Delta z$ [m]')
##    if case==0 or case==4:
##        ax_eps_lasso.set_ylabel(r'$D_{11}$ [m2/s2]', color=color)
##    last=len(deltaz_lasso)
##    ax_eps_lasso.plot(deltaz_lasso[1:last],deltaz_variance_lasso[1:last],color=color)
##    ax_eps_lasso.plot(deltaz_lasso[1:last],deltaz_varsum_lasso[1:last],color=color,linestyle='dashed')
##    ax_eps_lasso.plot(deltaz_lasso[1:last],factor23*deltaz_lasso[1:last]**(2/3),color='k',linestyle='dotted')
##    ax_eps_lasso.tick_params(axis='y', labelcolor=color)
##
##    ax_eps2_lasso = ax_eps_lasso.twinx()
##    color = 'tab:blue'
##    if case==3 or case==7:
##        ax_eps2_lasso.set_ylabel(r'$\epsilon$ [m2/s3]', color=color) 
##    ax_eps2_lasso.plot(deltaz_lasso[1:last],epsilon_lasso[1:last],color=color)
##    ax_eps2_lasso.tick_params(axis='y', labelcolor=color)
##    ax_eps2_lasso.ticklabel_format(scilimits=[-3,3])
##    ax_eps_lasso.set_title(silhs_files[case])
##
##
##    #figure showing D_11 and epsilon SILHS
##    ax_eps_silhs=fig_eps_silhs.add_subplot(2,4,case+1)
##    color = 'tab:red'
##    if case>3:
##        ax_eps_silhs.set_xlabel(r'$\Delta z$ [m]')
##    if case==0 or case==4:
##        ax_eps_silhs.set_ylabel(r'$D_{11}$ [m2/s2]', color=color)
##    last=len(deltaz_silhs)
##    ax_eps_silhs.plot(deltaz_silhs[1:last],deltaz_variance_silhs[1:last],color=color)
##    ax_eps_silhs.plot(deltaz_silhs[1:last],deltaz_varsum_silhs[1:last],color=color,linestyle='dashed')
##    ax_eps_silhs.plot(deltaz_silhs[1:last],deltaz_varmult_silhs[1:last],color=color,linestyle='dotted')
##    ax_eps_silhs.plot(deltaz_silhs[1:last],factor23*deltaz_silhs[1:last]**(2/3),color='k',linestyle='dotted')
##    ax_eps_silhs.tick_params(axis='y', labelcolor=color)
##
##    ax_eps2_silhs = ax_eps_silhs.twinx()
##    color = 'tab:blue'
##    if case==3 or case==7:
##        ax_eps2_silhs.set_ylabel(r'$\epsilon$ [m2/s3]', color=color)
##    ax_eps2_silhs.plot(deltaz_silhs[1:last],epsilon_silhs[1:last],color=color)
##    ax_eps2_silhs.tick_params(axis='y', labelcolor=color)
##    ax_eps2_silhs.ticklabel_format(scilimits=[-3,3])
##    ax_eps_silhs.set_title(silhs_files[case])
##
##
##    #figure showing D_11 for lidar, LASSO, and silhs
##    ax_d11=fig_d11.add_subplot(2,4,case+1)
##    if case>3:
##        ax_d11.set_xlabel(r'$\Delta z$ [m]')
##    if case==0 or case==4:
##        ax_d11.set_ylabel(r'$D_{11}$ [m2/s2]')
##    last=len(deltaz_dlraw)
##    ax_d11.plot(deltaz_dlraw[1:last],deltaz_variance_dlraw[1:last],label='Lidar')
##    last=len(deltaz_lasso)
##    ax_d11.plot(deltaz_lasso[1:last],deltaz_variance_lasso[1:last],label='LASSO')
##    last=len(deltaz_silhs)
##    ax_d11.plot(deltaz_silhs[1:last],deltaz_variance_silhs[1:last],label='SILHS')
##    ax_d11.plot(deltaz_lasso[1:last],0.025*deltaz_lasso[1:last]**(2/3),color='k',linestyle='dotted')
##    if case==3:
##        ax_d11.legend(frameon=False)
##    ax_d11.set_title(silhs_files[case])
#
#
#    #figure showing D_11, D_22, and D_33
#    ax_d11_d22_d33=fig_d11_d22_d33.add_subplot(2,4,case+1)
#    if case>3:
#        ax_d11_d22_d33.set_xlabel(r'$\Delta z$ [m]')
#    if case==0 or case==4:
#        ax_d11_d22_d33.set_ylabel(r'$D_{ij}$ [m2/s2]')
#    last=len(deltaz_lasso)
#    ax_d11_d22_d33.plot(deltaz_lasso[1:last],0.75*deltaz_variance_lasso_u[1:last],label=r'$\frac{3}{4}$D_11 (u)')
#    ax_d11_d22_d33.plot(deltaz_lasso[1:last],0.75*deltaz_variance_lasso_v[1:last],label=r'$\frac{3}{4}$D_22 (v)')
#    ax_d11_d22_d33.plot(deltaz_lasso[1:last],deltaz_variance_lasso[1:last],label='D_33 (w)')
#    ax_d11_d22_d33.plot(deltaz_lasso[1:last],0.025*deltaz_lasso[1:last]**(2/3),color='k',linestyle='dotted')
#    if case==3:
#        ax_d11_d22_d33.legend(frameon=False)
#    ax_d11_d22_d33.set_title(silhs_files[case])
#
#
#
    #figure showing corr(w(z+dz),w(z)) for lidar, LASSO, and silhs
    ax_d112=fig_d112.add_subplot(2,4,case+1)
    if case>3:
        ax_d112.set_xlabel(r'$\Delta z$ [m]')
    if case==0 or case==4:
        ax_d112.set_ylabel(r'corr$(w(z),w(z+\Delta z))$ [-]')
    last=len(deltaz_dlraw)
    ax_d112.plot(deltaz_dlraw[1:last],deltaz_varmult_dlraw[1:last],'b',label='Lidar (P)')
    ax_d112.plot(deltaz_dlraw[1:last],deltaz_varmult_dlraw2[1:last],'b--',label='Lidar (R)')
    last=len(deltaz_lasso)
    ax_d112.plot(deltaz_lasso[1:last],deltaz_varmult_lasso[1:last],'r',label='LES-UWM (P)')
    ax_d112.plot(deltaz_lasso[1:last],deltaz_varmult_lasso2[1:last],'r--',label='LES-UWM (R)')
    last=len(deltaz_silhs)
    ax_d112.plot(deltaz_silhs[1:last],deltaz_varmult_silhs[1:last],'g',label='SILHS, VDC={} (P)'.format(vdc1))
    ax_d112.plot(deltaz_silhs[1:last],deltaz_varmult_silhs2[1:last],'g--',label='SILHS, VDC={} (R)'.format(vdc1))
    ax_d112.plot(deltaz_silhs[1:last],deltaz_varmult_silhs30[1:last],'g-.',label='SILHS, VDC={} (P)'.format(vdc2))
    ax_d112.plot(deltaz_silhs[1:last],deltaz_varmult_silhs230[1:last],'g:',label='SILHS, VDC={} (R)'.format(vdc2))
    if case==3:
        ax_d112.legend(frameon=False,prop={'size': 7})
    ax_d112.set_title(silhs_files[case])



    #figure showing skew(w(z+dz),w(z)) for lidar, LASSO, and silhs
    ax_sciskew=fig_sciskew.add_subplot(2,4,case+1)
    if case>3:
        ax_sciskew.set_xlabel(r'$\Delta z$ [m]')
    if case==0 or case==4:
        ax_sciskew.set_ylabel(r'skewness$(\Delta w)$ [-]')
    last=len(deltaz_dlraw)
    ax_sciskew.plot(deltaz_dlraw[1:last],scistatskew_dlraw[1:last],'b',label='Lidar')
    last=len(deltaz_lasso)
    ax_sciskew.plot(deltaz_lasso[1:last],scistatskew_lasso[1:last],'r',label='LES-UWM')
    last=len(deltaz_silhs)
    ax_sciskew.plot(deltaz_silhs[1:last],scistatskew_silhs[1:last],'g',label='SILHS, VDC={}'.format(vdc1))
    ax_sciskew.plot(deltaz_silhs[1:last],scistatskew_silhs30[1:last],'g-.',label='SILHS, VDC={}'.format(vdc2))
    if case==3:
        ax_sciskew.legend(frameon=False,prop={'size': 7})
    ax_sciskew.set_title(silhs_files[case])
#


    #figure showing skew(w(z+dz),w(z)) for lidar, LASSO, and silhs
    ax_scikurt=fig_scikurt.add_subplot(2,4,case+1)
    if case>3:
        ax_scikurt.set_xlabel(r'$\Delta z$ [m]')
    if case==0 or case==4:
        ax_scikurt.set_ylabel(r'kurtosis$(\Delta w)$ [-]')
    last=len(deltaz_dlraw)
    ax_scikurt.plot(deltaz_dlraw[1:last],scistatkurt_dlraw[1:last],'b',label='Lidar')
    last=len(deltaz_lasso)
    ax_scikurt.plot(deltaz_lasso[1:last],scistatkurt_lasso[1:last],'r',label='LES-UWM')
    last=len(deltaz_silhs)
    ax_scikurt.plot(deltaz_silhs[1:last],scistatkurt_silhs[1:last],'g',label='SILHS, VDC={}'.format(vdc1))
    ax_scikurt.plot(deltaz_silhs[1:last],scistatkurt_silhs30[1:last],'g-.',label='SILHS, VDC={}'.format(vdc2))
    if case==3:
        ax_scikurt.legend(frameon=False,prop={'size': 7})
    ax_scikurt.set_title(silhs_files[case])
#
#
#
#    # scatter plot with average (nanmean) values of epsilon
#    ax_eps_scatter=fig_eps_scatter.add_subplot()
#    ax_eps_scatter.scatter(np.nanmean(epsilon_dlraw),np.nanmean(epsilon_lasso),c='b',label='DL vs.~LASSO')
#    ax_eps_scatter.scatter(np.nanmean(epsilon_dlraw),np.nanmean(epsilon_silhs),c='r',label='DL vs.~SILHS')
#    ax_eps_scatter.scatter(np.nanmean(epsilon_dlraw),em_dp1_mean,c='g')
##    ax_eps_scatter.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
##    ax_eps_scatter.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    axes=ax_eps_scatter.axis()
#    if case==0:
#        axes_x_min=axes[0]
#        axes_x_max=axes[1]
#        axes_y_min=axes[2]
#        axes_y_max=axes[3]
#    if case>0:
#        if axes[0]<axes_x_min:
#            axes_x_min=axes[0]
#        if axes[1]>axes_x_max:
#            axes_x_max=axes[1]
#        if axes[2]<axes_y_min:
#            axes_y_min=axes[2]
#        if axes[3]>axes_y_max:
#            axes_y_max=axes[3]
#    if case==7:
#        linex=np.array([axes_x_min,axes_x_max])
#        liney=np.array([axes_y_min,axes_y_min+(axes_x_max-axes_x_min)])
#        ax_eps_scatter.plot(linex,liney,'k--')
#    ax_eps_scatter.set_xlabel(r'Lidar $\left\langle\varepsilon\right\rangle$ [m$^2$s$^{-3}$]')
#    ax_eps_scatter.set_ylabel(r'$\left\langle\varepsilon\right\rangle$ [m$^2$s$^{-3}$]')
#
#
#
    # scatter plot with w----lidar
#    level1=400
#    level2=650
#    lidar_z_end_array=level2*np.ones(len(dl_raw_z))
#    lidar_z_start_array=level1*np.ones(len(dl_raw_z))
#    w_scatter_lidar_zsi=np.argmin((lidar_z_start_array-dl_raw_z)**2)
#    w_scatter_lidar_zei=np.argmin((lidar_z_end_array-dl_raw_z)**2)
#    ax_w_dl_scatter=fig_w_dl_scatter.add_subplot(2,4,case+1)
#    ax_w_dl_scatter.scatter(dl_raw_w[dl_raw_tsi:dl_raw_tei,w_scatter_lidar_zsi],dl_raw_w[dl_raw_tsi:dl_raw_tei,w_scatter_lidar_zei])
##    ax_w_dl_scatter.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
##    ax_w_dl_scatter.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    axes=ax_w_dl_scatter.axis()
#    linex4=np.array([axes[0],axes[1]])
#    liney4=np.array([axes[2],axes[2]+(axes[1]-axes[0])])
#    ax_w_dl_scatter.plot(linex4,liney4,'k--')
#    if case>3:
#        ax_w_dl_scatter.set_xlabel(r'Lidar $w$ [ms$^{-1}$] at 400 m')
#    if case==0 or case==4:
#        ax_w_dl_scatter.set_ylabel(r'Lidar $w$ [ms$^{-1}$] at 650 m')
#    ax_w_dl_scatter.set_title(silhs_files[case])


   # scatter plot with w---lasso
#    level1=400
#    level2=650
#    lasso_z_end_array=level2*np.ones(len(lasso_1m_z))
#    lasso_z_start_array=level1*np.ones(len(lasso_1m_z))
#    w_scatter_lasso_zsi=np.argmin((lasso_z_start_array-lasso_1m_z)**2)
#    w_scatter_lasso_zei=np.argmin((lasso_z_end_array-lasso_1m_z)**2)
#    ax_w_lasso_scatter=fig_w_lasso_scatter.add_subplot(2,4,case+1)
#    ax_w_lasso_scatter.scatter(lasso_1m_w[:,w_scatter_lasso_zsi],lasso_1m_w[:,w_scatter_lasso_zei])
##    ax_w_lasso_scatter.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
##    ax_w_lasso_scatter.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    axes=ax_w_lasso_scatter.axis()
#    linex5=np.array([axes[0],axes[1]])
#    liney5=np.array([axes[2],axes[2]+(axes[1]-axes[0])])
#    ax_w_lasso_scatter.plot(linex5,liney5,'k--')
#    if case>3:
#        ax_w_lasso_scatter.set_xlabel(r'LES-UWM $w$ [ms$^{-1}$] at 400 m')
#    if case==0 or case==4:
#        ax_w_lasso_scatter.set_ylabel(r'LES-UWM $w$ [ms$^{-1}$] at 650 m')
#    ax_w_lasso_scatter.set_title(silhs_files[case])
#

    # scatter plot with w SILHS
    level1=400
    level2=650
    silhs_z_end_array=level2*np.ones(len(silhs_z))
    silhs_z_start_array=level1*np.ones(len(silhs_z))
    w_scatter_silhs_zsi=np.argmin((silhs_z_start_array-silhs_z)**2)
    w_scatter_silhs_zei=np.argmin((silhs_z_end_array-silhs_z)**2)
    ax_w_scatter=fig_w_scatter.add_subplot(2,4,case+1)
    ax_w_scatter.scatter(silhs_w30[silhs_tsi:silhs_tei,w_scatter_silhs_zsi,0],silhs_w30[silhs_tsi:silhs_tei,w_scatter_silhs_zei,0],color='r',alpha=0.5,label='VDC={}'.format(vdc2)) 
    ax_w_scatter.scatter(silhs_w[silhs_tsi:silhs_tei,w_scatter_silhs_zsi,0],silhs_w[silhs_tsi:silhs_tei,w_scatter_silhs_zei,0],color='b',alpha=0.5,label='VDC={}'.format(vdc1))
#    ax_w_scatter.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#    ax_w_scatter.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes=ax_w_scatter.axis()
    linex2=np.array([axes[0],axes[1]])
    liney2=np.array([axes[2],axes[2]+(axes[1]-axes[0])])
    ax_w_scatter.plot(linex2,liney2,'k--')
    if case>3:
        ax_w_scatter.set_xlabel(r'SILHS $w$ [ms$^{-1}$] at 400 m')
    if case==0 or case==4:
        ax_w_scatter.set_ylabel(r'SILHS $w$ [ms$^{-1}$] at 650 m')
    if case==3:
        ax_w_scatter.legend(frameon=False)
    ax_w_scatter.set_title(silhs_files[case])


#    ax_w30_scatter=fig_w30_scatter.add_subplot(2,4,case+1)
#    ax_w30_scatter.scatter(silhs_w30[silhs_tsi:silhs_tei,w_scatter_silhs_zsi,0],silhs_w[silhs_tsi:silhs_tei,w_scatter_silhs_zei,0])
##    ax_w_scatter.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
##    ax_w_scatter.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    axes=ax_w30_scatter.axis()
#    linex2=np.array([axes[0],axes[1]])
#    liney2=np.array([axes[2],axes[2]+(axes[1]-axes[0])])
#    ax_w30_scatter.plot(linex2,liney2,'k--')
#    if case>3:
#        ax_w30_scatter.set_xlabel(r'SILHS $w$ [ms$^{-1}$] at 400 m')
#    if case==0 or case==4:
#        ax_w30_scatter.set_ylabel(r'SILHS $w$ [ms$^{-1}$] at 650 m')
#    ax_w30_scatter.set_title(silhs_files[case])
#



    # scatter plot with chi
    ax_chi_scatter=fig_chi_scatter.add_subplot(2,4,case+1)
    #ax_chi_scatter.scatter(silhs_chi_concat[:,w_scatter_silhs_zsi],silhs_chi_concat[:,w_scatter_silhs_zei])
    ax_chi_scatter.scatter(silhs_chi[silhs_tsi:silhs_tei,w_scatter_silhs_zsi,0],silhs_chi[silhs_tsi:silhs_tei,w_scatter_silhs_zei,0])
    ax_chi_scatter.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax_chi_scatter.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes=ax_chi_scatter.axis()
    linex3=np.array([axes[0],axes[1]])
    liney3=np.array([axes[2],axes[2]+(axes[1]-axes[0])])
    ax_chi_scatter.plot(linex3,liney3,'k--')
    if case>3:
        ax_chi_scatter.set_xlabel(r'$\chi$ [kg kg$^{-1}$]'+' at {} m'.format(level1))
    if case==0 or case==4:
        ax_chi_scatter.set_ylabel(r'$\chi$ [kg kg$^{-1}$]'+' at {} m'.format(level2))
    ax_chi_scatter.set_title(silhs_files[case])


#    rows=4
#    columns=8
#
#    # big plot with lidar and LASSO wp2 including time-height plots
#    ax_big_wp2=fig_big_wp2.add_subplot(rows,columns,(2*(case+1)-1,2*(case+1)-1+8) if case<4 else (2*((case+1)+4)-1,2*((case+1)+4)-1+8))
##    ax_big_wp2.plot(np.mean(lasso_w2[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label='LASSO',color='b')
#    ax_big_wp2.plot(np.mean(lasso_1m_w2[:,lasso_1m_zsi:lasso_1m_zei],0),lasso_1m_z[lasso_1m_zsi:lasso_1m_zei],label='LES-UWM',color='b')
#    ax_big_wp2.plot(np.mean(dl_w2[dl_tsi:dl_tei,dl_zsi:dl_zei],0),dl_z[dl_zsi:dl_zei],label='Lidar VAP',color='r')
#    if case>3:
#        ax_big_wp2.set_xlabel(r"$\overline{w'^2}$")
#    ax_big_wp2.set_ylabel('height [m]')
#    ax_big_wp2.set_title(silhs_files[case])
#    ax_big_wp2.legend(frameon=False)
#
#    ax_big_wp2=fig_big_wp2.add_subplot(rows,columns,2*(case+1) if case<4 else 2*((case+1)+4))
##    im=ax_big_wp2.imshow(lasso_w2[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei].T,origin='lower',cmap='rainbow',extent=[dl_t[dl_tsi],dl_t[dl_tei],dl_z[dl_zsi],dl_z[dl_zei_special]],aspect=3)
#    im=ax_big_wp2.imshow(lasso_1m_w2[:,lasso_1m_zsi:lasso_1m_zei].T,origin='lower',cmap='rainbow',extent=[dl_t[dl_tsi],dl_t[dl_tei],dl_z[dl_zsi],dl_z[dl_zei_special]],aspect=3)
#    fig_big_wp2.colorbar(im,ax=ax_big_wp2)
#    ax_big_wp2.set_ylabel('height [m]')
#    im.set_clim(0,2)
#    ax_big_wp2.set_title(r"LES-UWM $\overline{w'^2}$",color='b')
#    ax_big_wp2=fig_big_wp2.add_subplot(rows,columns,2*(case+1)+8 if case<4 else 2*((case+1)+4)+8)
#    im=ax_big_wp2.imshow(dl_w2[dl_tsi:dl_tei,dl_zsi:dl_zei_special].T,origin='lower',cmap='rainbow',extent=[dl_t[dl_tsi],dl_t[dl_tei],dl_z[dl_zsi],dl_z[dl_zei_special]],aspect=3)
#    fig_big_wp2.colorbar(im,ax=ax_big_wp2)
#    if case>3:
#        ax_big_wp2.set_xlabel('time [sec]')
#    ax_big_wp2.set_ylabel('height [m]')
#    ax_big_wp2.set_title(r"Lidar $\overline{w'^2}$",color='r')
#    im.set_clim(0,2)
#
#
#    # big plot with lidar and LASSO wp3 including time-height plots
#    ax_big_wp3=fig_big_wp3.add_subplot(rows,columns,(2*(case+1)-1,2*(case+1)-1+8) if case<4 else (2*((case+1)+4)-1,2*((case+1)+4)-1+8))
##    ax_big_wp3.plot(np.mean(lasso_w3[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label='LASSO',color='b')
#    ax_big_wp3.plot(np.mean(lasso_1m_w3[:,lasso_1m_zsi:lasso_1m_zei],0),lasso_1m_z[lasso_1m_zsi:lasso_1m_zei],label='LES-UWM',color='b')
#    ax_big_wp3.plot(np.mean(dl_w3[dl_tsi:dl_tei,dl_zsi:dl_zei],0),dl_z[dl_zsi:dl_zei],label='Lidar VAP',color='r')
#    if case>3:
#        ax_big_wp3.set_xlabel(r"$\overline{w'^3}$")
#    ax_big_wp3.set_ylabel('height [m]')
#    ax_big_wp3.set_title(silhs_files[case])
#    ax_big_wp3.legend(frameon=False)
#
#    ax_big_wp3=fig_big_wp3.add_subplot(rows,columns,2*(case+1) if case<4 else 2*((case+1)+4))
##    im3=ax_big_wp3.imshow(lasso_w3[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei].T,origin='lower',cmap='rainbow',extent=[dl_t[dl_tsi],dl_t[dl_tei],dl_z[dl_zsi],dl_z[dl_zei_special]],aspect=3)
#    im3=ax_big_wp3.imshow(lasso_1m_w3[:,lasso_1m_zsi:lasso_1m_zei].T,origin='lower',cmap='rainbow',extent=[dl_t[dl_tsi],dl_t[dl_tei],dl_z[dl_zsi],dl_z[dl_zei_special]],aspect=3)
#    fig_big_wp3.colorbar(im3,ax=ax_big_wp3)
#    ax_big_wp3.set_ylabel('height [m]')
#    im3.set_clim(-0.5,3)
#    ax_big_wp3.set_title(r"LES-UWM $\overline{w'^3}$",color='b')
#    ax_big_wp3=fig_big_wp3.add_subplot(rows,columns,2*(case+1)+8 if case<4 else 2*((case+1)+4)+8)
#    im3=ax_big_wp3.imshow(dl_w3[dl_tsi:dl_tei,dl_zsi:dl_zei_special].T,origin='lower',cmap='rainbow',extent=[dl_t[dl_tsi],dl_t[dl_tei],dl_z[dl_zsi],dl_z[dl_zei_special]],aspect=3)
#    fig_big_wp3.colorbar(im3,ax=ax_big_wp3)
#    if case>3:
#        ax_big_wp3.set_xlabel('time [sec]')
#    ax_big_wp3.set_ylabel('height [m]')
#    ax_big_wp3.set_title(r"Lidar $\overline{w'^3}$",color='r')
#    im3.set_clim(-0.5,3)
#
#
#   # lidar and LASSO w time-height plots
#    ax_big_w=fig_big_w.add_subplot(4,4,case+1 if case<4 else case+5)
#    im2=ax_big_w.imshow(lasso_1m_w.T,origin='lower',cmap='rainbow',extent=[dl_raw_t[dl_raw_tsi],dl_raw_t[dl_raw_tei],dl_raw_z[dl_raw_zsi],dl_raw_z[dl_raw_zei_special]],aspect=3)
#    cbar=fig_big_w.colorbar(im2,ax=ax_big_w)
#    cbar.set_label(r'LES-UWM $w$ [m/s]')
#    ax_big_w.set_ylabel('height [m]')
#    im2.set_clim(-3,5)
#    ax_big_w.set_title(silhs_files[case])
#    axes=ax_big_w.axis()
#    #ax_big_w.text(-0.01*(axes[1]-axes[0]),-0.2*(axes[3]-axes[2]),silhs_files[case],rotation=90,fontweight='bold')
#    ax_big_w=fig_big_w.add_subplot(4,4,case+5 if case<4 else case+9)
#    im2=ax_big_w.imshow(dl_raw_w[dl_raw_tsi:dl_raw_tei,dl_raw_zsi:dl_raw_zei_special].T,origin='lower',cmap='rainbow',extent=[dl_raw_t[dl_raw_tsi],dl_raw_t[dl_raw_tei],dl_raw_z[dl_raw_zsi],dl_raw_z[dl_raw_zei_special]],aspect=3)
#    cbar=fig_big_w.colorbar(im2,ax=ax_big_w)
#    cbar.set_label(r'Lidar $w$ [m/s]')
#    if case>3:
#        ax_big_w.set_xlabel('time [sec]')
#    ax_big_w.set_ylabel('height [m]')
#    #ax_big_w.set_title(r"Lidar $w$")
#    im2.set_clim(-3,5)



    # SILHS w histogram
    ax_w_hist=fig_w_hist.add_subplot(2,4,case+1)
    ax_w_hist.hist(silhs_w[silhs_tsi:silhs_tei,w_scatter_silhs_zsi,0],bins=15,alpha=0.5,color='b',label='400 m')
    ax_w_hist.hist(silhs_w[silhs_tsi:silhs_tei,w_scatter_silhs_zei,0],bins=15,alpha=0.5,color='r',label='650 m')
    if case>3:
        ax_w_hist.set_xlabel(r'SILHS $w$ [ms$^{-1}$]')
    if case==0 or case==4:
        ax_w_hist.set_ylabel('count [-]')
    if case==3:
        ax_w_hist.legend(frameon=False)
    ax_w_hist.set_title(silhs_files[case])    


    # SILHS chi histogram
    ax_chi_hist=fig_chi_hist.add_subplot(2,4,case+1)
    ax_chi_hist.hist(silhs_chi[silhs_tsi:silhs_tei,w_scatter_silhs_zsi,0],bins=15,alpha=0.5,color='b',label='400 m')
    ax_chi_hist.hist(silhs_chi[silhs_tsi:silhs_tei,w_scatter_silhs_zei,0],bins=15,alpha=0.5,color='r',label='650 m')
    ax_chi_hist.ticklabel_format(style='sci', axis='x', scilimits=(-2,2))
    if case>3:
        ax_chi_hist.set_xlabel(r'SILHS $\chi$ [kg kg$^{-1}$]')
    if case==0 or case==4:
        ax_chi_hist.set_ylabel('count [-]')
    if case==3:
        ax_chi_hist.legend(frameon=False)
    ax_chi_hist.set_title(silhs_files[case])



#bar chart with correlation lengths
x_coords=np.array([1,2,3,4,5,6,7,8])
ax_bar=fig_bar.add_subplot()
ax_bar.bar(x_coords-0.2,integral_dl,0.2,label='Lidar')
ax_bar.bar(x_coords,integral_lasso,0.2,label='LES-UWM')
ax_bar.bar(x_coords+0.2,integral_silhs,0.2,label='SILHS')
ax_bar.set_ylim(0,800)
ax_bar.set_xticks(x_coords)
ax_bar.set_xticklabels(silhs_files,rotation=45)
ax_bar.set_ylabel('Correlation length [m]')
ax_bar.legend(frameon=False)








## output the figures
fig_dl_raw_w.savefig('figs/dl_raw_w.png',dpi=300,bbox_inches="tight")
fig_dl_raw_w2.savefig('figs/dl_raw_w2.png',dpi=300,bbox_inches="tight")

#fig_iso.subplots_adjust(hspace=0.325)
#fig_iso.savefig('figs/iso.png',dpi=300,bbox_inches="tight")

fig_wp2.savefig('figs/wp2.png',dpi=300,bbox_inches="tight")
fig_wp3.savefig('figs/wp3.png',dpi=300,bbox_inches="tight")
#fig_wskew.savefig('figs/wskew.png',dpi=300,bbox_inches="tight")
#fig_d11.savefig('figs/d11.png',dpi=300,bbox_inches="tight")
fig_d112.savefig('figs/d112.png',dpi=300,bbox_inches="tight")

#fig_eps.subplots_adjust(hspace=0.325)
#fig_eps.subplots_adjust(wspace=0.5)
#fig_eps.savefig('figs/eps.png',dpi=300,bbox_inches='tight')

#fig_eps_lasso.subplots_adjust(hspace=0.325)
#fig_eps_lasso.subplots_adjust(wspace=0.5)
#fig_eps_lasso.savefig('figs/eps_lasso.png',dpi=300,bbox_inches='tight')

#fig_eps_silhs.subplots_adjust(hspace=0.325)
#fig_eps_silhs.subplots_adjust(wspace=0.5)
#fig_eps_silhs.savefig('figs/eps_silhs.png',dpi=300,bbox_inches='tight')

#fig_eps_scatter.savefig('figs/eps_scatter.png',dpi=300,bbox_inches='tight')

#fig_big_wp2.savefig('figs/big_wp2.png',dpi=300,bbox_inches='tight')
#fig_big_wp3.savefig('figs/big_wp3.png',dpi=300,bbox_inches='tight')

#fig_big_w.subplots_adjust(wspace=0.5)
#fig_big_w.savefig('figs/big_w.png',dpi=300,bbox_inches='tight')

#fig_d11_d22_d33.savefig('figs/d11_d22_d33.png',dpi=300,bbox_inches='tight')

fig_u_v.savefig('figs/u_v.png',dpi=300,bbox_inches='tight')
fig_chi.savefig('figs/chi.png',dpi=300,bbox_inches='tight')
fig_w_scatter.savefig('figs/w_scatter.png',dpi=300,bbox_inches='tight')
#fig_w30_scatter.savefig('figs/w30_scatter.png',dpi=300,bbox_inches='tight') 
fig_w_dl_scatter.savefig('figs/w_dl_scatter.png',dpi=300,bbox_inches='tight')
fig_w_lasso_scatter.savefig('figs/w_lasso_scatter.png',dpi=300,bbox_inches='tight')
fig_chi_scatter.savefig('figs/chi_scatter.png',dpi=300,bbox_inches='tight')
fig_w_hist.savefig('figs/w_hist.png',dpi=300,bbox_inches='tight')
fig_chi_hist.savefig('figs/chi_hist.png',dpi=300,bbox_inches='tight')
fig_mf.savefig('figs/mf.png',dpi=300,bbox_inches='tight')
fig_bar.savefig('figs/bar.png',dpi=300,bbox_inches='tight')

fig_sciskew.savefig('figs/sciskew.png',dpi=300,bbox_inches='tight')
fig_scikurt.savefig('figs/scikurt.png',dpi=300,bbox_inches='tight')

