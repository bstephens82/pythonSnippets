#master script for comparing lidar (raw and stats), lasso, & silhs

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

def calculate_variance_epsilon(z,w):

    base_level=3
    num_points=30
    deltaz=np.zeros((num_points))
    deltaz_variance=np.zeros((num_points))
    epsilon=np.zeros((num_points))
    spacer=1
    for k in range(0,num_points):
        deltaz[k]=z[base_level+spacer*k]-z[base_level]
        deltaz_variance[k]=np.var(w[:,base_level+spacer*k]-w[:,base_level],0,ddof=1)
        if k==0:
            epsilon[k]=np.nan
        if k>0:
            epsilon[k]=deltaz_variance[k]**(3/2)/(2.**(3/2))/deltaz[k]

    return deltaz, deltaz_variance, epsilon

#where all the data is located
dl_folder='/glade/work/stepheba/lidar/'
dl_stats=['dl20170627.nc','dl20170717.nc','dl20170728.nc','dl20170923.nc','dl20180911.nc','dl20180917.nc','dl20180918.nc','dl20181002.nc']
dl_raw=['dl20170627_raw.nc','dl20170717_raw.nc','dl20170728_raw.nc','dl20170923_raw.nc','dl20180911_raw.nc','dl20180917_raw.nc','dl20180918_raw.nc','dl20181002_raw.nc']

lasso_folder='/glade/work/stepheba/lasso_orig/'
lasso_les=['2017-06-27','2017-07-17','2017-07-28','2017-09-23/02','2018-09-11/08','2018-09-17/08','2018-09-18/08','2018-10-02/08']

lasso_1m_folder='/glade/work/stepheba/lasso_1_min_output/'
lasso_1m_les=['2017-06-27','2017-07-17','2017-07-28','2017-09-23','2018-09-11','2018-09-17','2018-09-18','2018-10-02']

silhs_folder0='/glade/work/stepheba/lasso_silhs/lasso_job_vdc_0/'
silhs_folder01='/glade/work/stepheba/lasso_silhs/lasso_job_vdc_01/'
silhs_files=['2017-06-27','2017-07-17','2017-07-28','2017-09-23','2018-09-11','2018-09-17','2018-09-18','2018-10-02']


# finding the right indices for time and height for all data (i.e. DL,
# DL raw, LASSO, WRF-CLUBB-SILHS)

# ADJUST ------------------------------
hour_start=17  #min should be 12 since LASSO and SILHS effectively start at 12
hour_end=22  #max should be 27 since LASSO and SILHS effectively end at 27

height_start=0
height_end=4000
# -------------------------------------

second_start=hour_start*3600
second_end=hour_end*3600
minute_start=hour_start*60
minute_end=hour_end*60
lasso_minute_start=(hour_start-12)*60
lasso_minute_end=(hour_end-12)*60
silhs_second_start=(hour_start-12)*3600
silhs_second_end=(hour_end-12)*3600

fig_dl_raw_w=plt.figure(figsize=(14.,7.))
fig_iso=plt.figure(figsize=(14.,7.))
fig_wp2=plt.figure(figsize=(14.,7.))
fig_wp3=plt.figure(figsize=(14.,7.))
fig_wskew=plt.figure(figsize=(14.,7.))
fig_eps=plt.figure(figsize=(14.,7.))

for case in range(0,8):
    #load the datasets
    dl_raw_data=Dataset(dl_folder+dl_raw[case])
    dl_stats_data=Dataset(dl_folder+dl_stats[case])
    lasso_data=Dataset(lasso_folder+lasso_les[case]+'/raw_model/'+'wrfstat_d01_'+lasso_les[case][0:10]+'_12:00:00.nc')
    lasso_1m_stats_data=Dataset(lasso_1m_folder+lasso_1m_les[case]+'/wrfstat.nc')
    lasso_1m_dflt_data=Dataset(lasso_1m_folder+lasso_1m_les[case]+'/wrfout.nc')
    silhs_data=Dataset(silhs_folder0+silhs_files[case]+'/lasso_'+silhs_files[case]+'_nl_lh_sample_points_2D.nc')
    clubb_zm_data=Dataset(silhs_folder0+silhs_files[case]+'/lasso_'+silhs_files[case]+'_zm_wrf.nc')
    clubb_zt_data=Dataset(silhs_folder0+silhs_files[case]+'/lasso_'+silhs_files[case]+'_zt_wrf.nc')
    
    #extract time and height variables
       #doppler lidar raw
    dl_raw_t=np.array(dl_raw_data.variables['time'])
    dl_raw_z=np.array(dl_raw_data.variables['range'])
    dl_raw_w=np.array(dl_raw_data.variables['radial_velocity'])
    dl_raw_int=np.array(dl_raw_data.variables['intensity'])
    dl_raw_snr=dl_raw_int-1
    
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
    lasso_w=np.array(lasso_data.variables['CSV_W'])
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
    clubb_wp2=np.mean(np.array(clubb_zm_data.variables['wp2']),(2,3))
    clubb_wp3=np.mean(np.array(clubb_zt_data.variables['wp3']),(2,3))
    clubb_wskew=np.mean(np.array(clubb_zt_data.variables['Skw_zt']),(2,3))
    
    #process the time and height data to reflect the desired frame
      #doppler lidar processed stats
    dl_t_start_array=second_start*np.ones((len(dl_t)))
    dl_t_end_array=second_end*np.ones(len(dl_t))
    dl_tsi=np.argmin((dl_t_start_array-dl_t)**2)
    dl_tei=np.argmin((dl_t_end_array-dl_t)**2)
    print('DL processed stats, time:')
    print(dl_t[dl_tsi])
    print(dl_t[dl_tei])
    
    dl_z_start_array=height_start*np.ones((len(dl_z)))
    dl_z_end_array=height_end*np.ones(len(dl_z))
    dl_zsi=np.argmin((dl_z_start_array-dl_z)**2)
    dl_zei=np.argmin((dl_z_end_array-dl_z)**2)
    print('height:')
    print(dl_z[dl_zsi])
    print(dl_z[dl_zei])
    
       #doppler lidar raw data
    dl_raw_t_start_array=second_start*np.ones((len(dl_raw_t)))
    dl_raw_t_end_array=second_end*np.ones((len(dl_raw_t)))
    dl_raw_tsi=np.argmin((dl_raw_t_start_array-dl_raw_t)**2)
    dl_raw_tei=np.argmin((dl_raw_t_end_array-dl_raw_t)**2)
    print('DL raw, time:')
    print(dl_raw_t[dl_raw_tsi])
    print(dl_raw_t[dl_raw_tei])
    
    dl_raw_z_start_array=height_start*np.ones((len(dl_raw_z)))
    dl_raw_z_end_array=height_end*np.ones((len(dl_raw_z)))
    dl_raw_zsi=np.argmin((dl_raw_z_start_array-dl_raw_z)**2)
    dl_raw_zei=np.argmin((dl_raw_z_end_array-dl_raw_z)**2)
    print('height:')
    print(dl_raw_z[dl_raw_zsi])
    print(dl_raw_z[dl_raw_zei])
    
       #original LASSO
    lasso_t_start_array=lasso_minute_start*np.ones((len(lasso_t)))
    lasso_t_end_array=lasso_minute_end*np.ones(len(lasso_t))
    lasso_tsi=np.argmin((lasso_t_start_array-lasso_t)**2)
    lasso_tei=np.argmin((lasso_t_end_array-lasso_t)**2)
    print('LASSO orig, time:')
    print(lasso_t[lasso_tsi])
    print(lasso_t[lasso_tei])
    
    lasso_z_start_array=height_start*np.ones((len(lasso_z)))
    lasso_z_end_array=height_end*np.ones(len(lasso_z))
    lasso_zsi=np.argmin((lasso_z_start_array-lasso_z)**2)
    lasso_zei=np.argmin((lasso_z_end_array-lasso_z)**2)
    print('height:')
    print(lasso_z[lasso_zsi])
    print(lasso_z[lasso_zei])
    
       #lasso 1-minute output Cheyenne reproduction
    lasso_1m_t_start_array=lasso_minute_start*np.ones((len(lasso_1m_t)))
    lasso_1m_t_end_array=lasso_minute_end*np.ones(len(lasso_1m_t))
    lasso_1m_tsi=np.argmin((lasso_1m_t_start_array-lasso_1m_t)**2)
    lasso_1m_tei=np.argmin((lasso_1m_t_end_array-lasso_1m_t)**2)
    print('LASSO 1m, time:')
    print(lasso_1m_t[lasso_1m_tsi])
    print(lasso_1m_t[lasso_1m_tei])
    
    lasso_1m_z_start_array=height_start*np.ones((len(lasso_1m_z)))
    lasso_1m_z_end_array=height_end*np.ones(len(lasso_1m_z))
    lasso_1m_zsi=np.argmin((lasso_1m_z_start_array-lasso_1m_z)**2)
    lasso_1m_zei=np.argmin((lasso_1m_z_end_array-lasso_1m_z)**2)
    print('height:')
    print(lasso_1m_z[lasso_1m_zsi])
    print(lasso_1m_z[lasso_1m_zei])
    
       #wrf-clubb-silhs output
    silhs_t_start_array=silhs_second_start*np.ones((len(silhs_t)))
    silhs_t_end_array=silhs_second_end*np.ones(len(silhs_t))
    silhs_tsi=np.argmin((silhs_t_start_array-silhs_t)**2)
    silhs_tei=np.argmin((silhs_t_end_array-silhs_t)**2)
    print('SILHS orig, time:')
    print(silhs_t[silhs_tsi])
    print(silhs_t[silhs_tei])
    
    silhs_z_start_array=height_start*np.ones((len(silhs_z)))
    silhs_z_end_array=height_end*np.ones(len(silhs_z))
    silhs_zsi=np.argmin((silhs_z_start_array-silhs_z)**2)
    silhs_zei=np.argmin((silhs_z_end_array-silhs_z)**2)
    print('height:')
    print(silhs_z[silhs_zsi])
    print(silhs_z[silhs_zei])
  
    #load only the part of the 3d vert. velocity that is necessary since it 
    #takes up a lot of space
#    lasso_1m_w=np.array(lasso_1m_stats_data.variables['CSV_W'][lasso_1m_tsi:lasso_1m_tei,lasso_1m_zsi:lasso_1m_zei])
    lasso_1m_cf=np.mean(np.array(lasso_1m_dflt_data.variables['CLDFRA'][lasso_1m_tsi:lasso_1m_tei,lasso_1m_zsi:lasso_1m_zei,:,:]),(2,3)) 

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


    #for raw doppler lidar, no need to deal with horizontal or other dimensions
    deltaz_dlraw, deltaz_variance_dlraw, epsilon_dlraw = calculate_variance_epsilon(z=dl_raw_z,w=dl_raw_w[dl_raw_tsi:dl_raw_tei,:])

    #for lasso 1-minute output, need to choose columns within the grid
#    deltaz_lasso, deltaz_variance_lasso, epsilon_lasso = calculate_variance_epsilon(z=lasso_1m_z,w=lasso_1m_w[lasso_1m_tsi:lasso_1m_tei,:,:,:])

    #for silhs output, need to select subcolumns
#    deltaz_silhs, deltaz_variance_silhs, epsilon_silhs = calculate_variance_epsilon(z=silhs_z,w=silhs_w[silhs_tsi:silhs_tei,:])


    #figure with up2 vp2 wp2 and cloud frac
    ax_iso=fig_iso.add_subplot(2,4,case+1)
    ax_iso.plot(np.mean(lasso_u2[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label='up2')
    ax_iso.plot(np.mean(lasso_v2[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label='vp2')
    ax_iso.plot(np.mean(lasso_w2[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label='wp2')
#    ax_iso.plot(np.mean(lasso_1m_cf,0),lasso_1m_z[lasso_1m_zsi:lasso_1m_zei],label='cld_frc')
    if case > 3:
        ax_iso.set_xlabel('[m2/s2]')
    if case == 0 or case == 4:
        ax_iso.set_ylabel('height [m]')
    ax_iso2 = ax_iso.twiny()  # create a second axis that shares the same x-axis    
    color = 'tab:red'
    if case<4:
        ax_iso2.set_xlabel('cloud frac. [-]', color=color)  # we already handled the x-label with ax1
    ax_iso2.plot(np.mean(lasso_1m_cf,0),lasso_1m_z[lasso_1m_zsi:lasso_1m_zei],color=color)
    ax_iso2.tick_params(axis='x', labelcolor=color)
    ax_iso.set_title(silhs_files[case])
    ax_iso.legend()

    #dl_raw_w
    ax_dl_raw_w=fig_dl_raw_w.add_subplot(2,4,case+1)
    ax_dl_raw_w.plot(np.nanmean(dl_raw_w[dl_raw_tsi:dl_raw_tei,dl_raw_zsi:dl_raw_zei],0),dl_raw_z[dl_raw_zsi:dl_raw_zei],label='DL')
    #plt.imshow(np.mean(lasso_w,(2,3)).T,origin='lower')
    if case > 3:
        ax_dl_raw_w.set_xlabel('w [m/s]')
    if case == 0 or case == 4:
        ax_dl_raw_w.set_ylabel('height [m]')
    ax_dl_raw_w.set_title(silhs_files[case])

    #wp2
    ax1=fig_wp2.add_subplot(2,4,case+1)
    ax1.plot(np.nanmean(dl_w2[dl_tsi:dl_tei,dl_zsi:dl_zei],0),dl_z[dl_zsi:dl_zei],label='DL')
    ax1.plot(np.mean(lasso_w2[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label='LASSO')
    ax1.plot(np.mean(lasso_1m_w2[lasso_1m_tsi:lasso_1m_tei,lasso_1m_zsi:lasso_1m_zei],0),lasso_z[lasso_1m_zsi:lasso_1m_zei],label='LASSO 1m')
    ax1.plot(np.mean(clubb_wp2[silhs_tsi:silhs_tei,silhs_zsi:silhs_zei],0),silhs_z[silhs_zsi:silhs_zei],label='CLUBB-SILHS')
    #plt.imshow(np.mean(lasso_w,(2,3)).T,origin='lower')
    if case > 3:
        ax1.set_xlabel('wp2 [m2/s2]')
    if case == 0 or case == 4:
        ax1.set_ylabel('height [m]')
    ax1.set_title(silhs_files[case])
    ax1.legend()
    #plt.subplots_adjust(bottom=0.2)

    #wp3
    ax3=fig_wp3.add_subplot(2,4,case+1)
    ax3.plot(np.nanmean(dl_w3[dl_tsi:dl_tei,dl_zsi:dl_zei],0),dl_z[dl_zsi:dl_zei],label='DL')
    ax3.plot(np.mean(lasso_w3[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label='LASSO')
    ax3.plot(np.mean(lasso_1m_w3[lasso_1m_tsi:lasso_1m_tei,lasso_1m_zsi:lasso_1m_zei],0),lasso_z[lasso_1m_zsi:lasso_1m_zei],label='LASSO 1m')
    ax3.plot(np.mean(clubb_wp3[silhs_tsi:silhs_tei,silhs_zsi:silhs_zei],0),silhs_z[silhs_zsi:silhs_zei],label='CLUBB-SILHS')
    #plt.imshow(np.mean(lasso_w,(2,3)).T,origin='lower')
    if case > 3:
        ax3.set_xlabel('wp3 [m3/s3]')
    if case == 0 or case == 4:
        ax3.set_ylabel('height [m]')
    ax3.set_title(silhs_files[case])
    ax3.legend()
    #plt.subplots_adjust(bottom=0.2)

    #skewness
    ax2=fig_wskew.add_subplot(2,4,case+1)
    ax2.plot(np.nanmean(dl_wskew[dl_tsi:dl_tei,dl_zsi:dl_zei],0),dl_z[dl_zsi:dl_zei],label='DL')
    ax2.plot(np.mean(lasso_wskew[lasso_tsi:lasso_tei,lasso_zsi:lasso_zei],0),lasso_z[lasso_zsi:lasso_zei],label='LASSO')
    ax2.plot(np.mean(lasso_1m_wskew[lasso_1m_tsi:lasso_1m_tei,lasso_1m_zsi:lasso_1m_zei],0),lasso_z[lasso_1m_zsi:lasso_1m_zei],label='LASSO 1m')
    ax2.plot(np.mean(clubb_wskew[silhs_tsi:silhs_tei,silhs_zsi:silhs_zei],0),silhs_z[silhs_zsi:silhs_zei],label='CLUBB-SILHS')
    #plt.imshow(np.mean(lasso_w,(2,3)).T,origin='lower')
    if case > 3:
        ax2.set_xlabel('w_skewness [-]')
    if case == 0 or case == 4:
        ax2.set_ylabel('height [m]')
    ax2.set_title(silhs_files[case])
    ax2.legend()
    #plt.subplots_adjust(bottom=0.2)

    #figure showing D_11 and epsilon DL raw
    ax_eps=fig_eps.add_subplot(2,4,case+1)
    color = 'tab:red'
    if case>3:
        ax_eps.set_xlabel(r'$\Delta z$ [m]')
    if case==0 or case==4:
        ax_eps.set_ylabel(r'$D_{11}$ [m2/s2]', color=color)
    ax_eps.plot(deltaz_dlraw,deltaz_variance_dlraw,color=color)
    ax_eps.tick_params(axis='y', labelcolor=color)

    ax_eps2 = ax_eps.twinx()  # create a second axis that shares the same x-axis    
    color = 'tab:blue'
    if case==3 or case==7:
        ax_eps2.set_ylabel(r'$\epsilon$ [m2/s3]', color=color)  # we already handled the x-label with ax1
    ax_eps2.plot(deltaz_dlraw,epsilon_dlraw,color=color)
    ax_eps2.tick_params(axis='y', labelcolor=color)
    ax_eps2.ticklabel_format(scilimits=[-3,3])
    ax_eps.set_title(silhs_files[case])



# output the figures
fig_dl_raw_w.savefig('figs/dl_raw_w.png',dpi=300,bbox_inches="tight")
fig_iso.savefig('figs/iso.png',dpi=300,bbox_inches="tight")
fig_wp2.savefig('figs/wp2.png',dpi=300,bbox_inches="tight")
fig_wp3.savefig('figs/wp3.png',dpi=300,bbox_inches="tight")
fig_wskew.savefig('figs/wskew.png',dpi=300,bbox_inches="tight")

fig_eps.subplots_adjust(hspace=0.325)
fig_eps.subplots_adjust(wspace=0.5)
fig_eps.savefig('figs/eps.png',dpi=300,bbox_inches='tight')






#    dl_shape=dl_w.shape
#    for t in range(0,dl_shape[0]):
#        for z in range(0,dl_shape[1]):
#            if dl_noise[t,z]>1.0 or dl_snr[t,z]<0.008:
#                dl_w[t,z]=np.nan
#                dl_w2[t,z]=np.nan
#                dl_wskew[t,z]=np.nan
#            if abs(dl_w2[t,z])>100:
#                dl_w2[t,z]=np.nan
#            if abs(dl_wskew[t,z])>100:
#                dl_wskew=np.nan


#    dl_raw_shape=dl_raw_w.shape
#    for t in range(0,dl_raw_shape[0]):
#        for z in range(0,dl_raw_shape[1]):
#            if dl_raw_snr[t,z]<0.008:
#                dl_raw_w[t,z]=np.nan


#    base_level=3
#    num_points=30
#    deltaz=np.zeros((num_points))
#    deltaz_variance=np.zeros((num_points))
#    epsilon=np.zeros((num_points))
#    spacer=1
#    for k in range(0,num_points):
#        deltaz[k]=dl_raw_z[base_level+spacer*k]-dl_raw_z[base_level]
#        deltaz_variance[k]=np.cov(dl_raw_w[dl_raw_tsi:dl_raw_tei,base_level+spacer*k]-dl_raw_w[dl_raw_tsi:dl_raw_tei,base_level],dl_raw_w[dl_raw_tsi:dl_raw_tei,base_level+spacer*k]-dl_raw_w[dl_raw_tsi:dl_raw_tei,base_level])[0,1]
#        #deltaz_variance[k]=np.var(dl_raw_w[dl_raw_tsi:dl_raw_tei,base_level+spacer*k]-dl_raw_w[dl_raw_tsi:dl_raw_tei,base_level],0,ddof=1)
#        if k==0:
#            epsilon[k]=np.nan
#        if k>0:
#            epsilon[k]=deltaz_variance[k]**(3/2)/(2.**(3/2))/deltaz[k]

