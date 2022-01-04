#master script for comparing lidar (raw and stats), lasso, & silhs

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

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
height_end=2000
# -------------------------------------

second_start=hour_start*3600
second_end=hour_end*3600
minute_start=hour_start*60
minute_end=hour_end*60
lasso_minute_start=(hour_start-12)*60
lasso_minute_end=(hour_end-12)*60
silhs_second_start=(hour_start-12)*3600
silhs_second_end=(hour_end-12)*3600


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
    lasso_z=np.array(lasso_data.variables['CSP_Z'])
    lasso_z=np.mean(lasso_z,0)
    lasso_z8w=np.array(lasso_data.variables['CSP_Z8W'])
    lasso_z8w=np.mean(lasso_z8w,0)
    lasso_w=np.array(lasso_data.variables['CSV_W'])
    lasso_w2=np.array(lasso_data.variables['CSP_W2'])
    lasso_wskew=np.array(lasso_data.variables['CSP_WSKEW'])
    
       #1-minute cheyenne LASSO LES reproduction
    lasso_1m_t=np.array(lasso_1m_dflt_data.variables['XTIME'])
    lasso_1m_z=np.array(lasso_1m_stats_data.variables['CSP_Z'])
    lasso_1m_z=np.mean(lasso_1m_z,0)
    lasso_1m_z8w=np.array(lasso_1m_stats_data.variables['CSP_Z8W'])
    lasso_1m_z8w=np.mean(lasso_1m_z8w,0)
    lasso_1m_w2=np.array(lasso_1m_stats_data.variables['CSP_W2'])
    lasso_1m_wskew=np.array(lasso_1m_stats_data.variables['CSP_WSKEW'])
    
       #wrf-clubb-silhs lasso coarse-grained runs
    silhs_t=np.array(silhs_data.variables['time'])
    silhs_z=np.array(silhs_data.variables['altitude'])
    silhs_w=np.mean(np.array(silhs_data.variables['w']),(2,3))
    clubb_wp2=np.mean(np.array(clubb_zm_data.variables['wp2']),(2,3))
    clubb_wskew=np.mean(np.array(clubb_zt_data.variables['Skw_zt']),(2,3))
    
    #process the time and height data to reflect the desired frame
      #doppler lidar processed stats
    dl_t_start_array=second_start*np.ones((len(dl_t)))
    dl_t_end_array=second_end*np.ones(len(dl_t))
    dl_t_start_idx=np.argmin((dl_t_start_array-dl_t)**2)
    dl_t_end_idx=np.argmin((dl_t_end_array-dl_t)**2)
    print(dl_t[dl_t_start_idx])
    print(dl_t[dl_t_end_idx])
    
    dl_z_start_array=height_start*np.ones((len(dl_z)))
    dl_z_end_array=height_end*np.ones(len(dl_z))
    dl_z_start_idx=np.argmin((dl_z_start_array-dl_z)**2)
    dl_z_end_idx=np.argmin((dl_z_end_array-dl_z)**2)
    print(dl_z[dl_z_start_idx])
    print(dl_z[dl_z_end_idx])
    
       #doppler lidar raw data
    dl_raw_t_start_array=second_start*np.ones((len(dl_raw_t)))
    dl_raw_t_end_array=second_end*np.ones((len(dl_raw_t)))
    dl_raw_t_start_idx=np.argmin((dl_raw_t_start_array-dl_raw_t)**2)
    dl_raw_t_end_idx=np.argmin((dl_raw_t_end_array-dl_raw_t)**2)
    print(dl_raw_t[dl_raw_t_start_idx])
    print(dl_raw_t[dl_raw_t_end_idx])
    
    dl_raw_z_start_array=height_start*np.ones((len(dl_raw_z)))
    dl_raw_z_end_array=height_end*np.ones((len(dl_raw_z)))
    dl_raw_z_start_idx=np.argmin((dl_raw_z_start_array-dl_raw_z)**2)
    dl_raw_z_end_idx=np.argmin((dl_raw_z_end_array-dl_raw_z)**2)
    print(dl_raw_z[dl_raw_z_start_idx])
    print(dl_raw_z[dl_raw_z_end_idx])
    
       #original LASSO
    lasso_t_start_array=lasso_minute_start*np.ones((len(lasso_t)))
    lasso_t_end_array=lasso_minute_end*np.ones(len(lasso_t))
    lasso_t_start_idx=np.argmin((lasso_t_start_array-lasso_t)**2)
    lasso_t_end_idx=np.argmin((lasso_t_end_array-lasso_t)**2)
    print(lasso_t[lasso_t_start_idx])
    print(lasso_t[lasso_t_end_idx])
    
    lasso_z_start_array=height_start*np.ones((len(lasso_z)))
    lasso_z_end_array=height_end*np.ones(len(lasso_z))
    lasso_z_start_idx=np.argmin((lasso_z_start_array-lasso_z)**2)
    lasso_z_end_idx=np.argmin((lasso_z_end_array-lasso_z)**2)
    print(lasso_z[lasso_z_start_idx])
    print(lasso_z[lasso_z_end_idx])
    
       #lasso 1-minute output Cheyenne reproduction
    lasso_1m_t_start_array=lasso_minute_start*np.ones((len(lasso_1m_t)))
    lasso_1m_t_end_array=lasso_minute_end*np.ones(len(lasso_1m_t))
    lasso_1m_t_start_idx=np.argmin((lasso_1m_t_start_array-lasso_1m_t)**2)
    lasso_1m_t_end_idx=np.argmin((lasso_1m_t_end_array-lasso_1m_t)**2)
    print(lasso_1m_t[lasso_1m_t_start_idx])
    print(lasso_1m_t[lasso_1m_t_end_idx])
    
    lasso_1m_z_start_array=height_start*np.ones((len(lasso_1m_z)))
    lasso_1m_z_end_array=height_end*np.ones(len(lasso_1m_z))
    lasso_1m_z_start_idx=np.argmin((lasso_1m_z_start_array-lasso_1m_z)**2)
    lasso_1m_z_end_idx=np.argmin((lasso_1m_z_end_array-lasso_1m_z)**2)
    print(lasso_1m_z[lasso_1m_z_start_idx])
    print(lasso_1m_z[lasso_1m_z_end_idx])
    
       #wrf-clubb-silhs output
    silhs_t_start_array=silhs_second_start*np.ones((len(silhs_t)))
    silhs_t_end_array=silhs_second_end*np.ones(len(silhs_t))
    silhs_t_start_idx=np.argmin((silhs_t_start_array-silhs_t)**2)
    silhs_t_end_idx=np.argmin((silhs_t_end_array-silhs_t)**2)
    print(silhs_t[silhs_t_start_idx])
    print(silhs_t[silhs_t_end_idx])
    
    silhs_z_start_array=height_start*np.ones((len(silhs_z)))
    silhs_z_end_array=height_end*np.ones(len(silhs_z))
    silhs_z_start_idx=np.argmin((silhs_z_start_array-silhs_z)**2)
    silhs_z_end_idx=np.argmin((silhs_z_end_array-silhs_z)**2)
    print(silhs_z[silhs_z_start_idx])
    print(silhs_z[silhs_z_end_idx])
    
    #load only the part of the 3d vert. velocity that is necessary since it 
    #takes up a lot of space
    lasso_1m_w=np.array(lasso_1m_stats_data.variables['CSV_W'][lasso_1m_t_start_idx:lasso_1m_t_end_idx,lasso_1m_z_start_idx:lasso_1m_z_end_idx])
    
    
    
    #process lidar stats variables
    dl_shape=dl_w.shape
    for t in range(0,dl_shape[0]):
        for z in range(0,dl_shape[1]):
            if dl_noise[t,z]>1.0 or dl_snr[t,z]<0.008:
                dl_w[t,z]=np.nan
                dl_w2[t,z]=np.nan
                dl_wskew[t,z]=np.nan
    
    #process lidar raw variables
    dl_raw_shape=dl_raw_w.shape
    for t in range(0,dl_raw_shape[0]):
        for z in range(0,dl_raw_shape[1]):
            if dl_raw_snr[t,z]<0.008:
                dl_raw_w[t,z]=np.nan


    plt.plot(np.nanmean(dl_w2[dl_t_start_idx:dl_t_end_idx,dl_z_start_idx:dl_z_end_idx],0),dl_z[dl_z_start_idx:dl_z_end_idx],label='DL')
    plt.plot(np.mean(lasso_w2[lasso_t_start_idx:lasso_t_end_idx,lasso_z_start_idx:lasso_z_end_idx],0),lasso_z[lasso_z_start_idx:lasso_z_end_idx],label='LASSO')
    plt.plot(np.mean(lasso_1m_w2[lasso_1m_t_start_idx:lasso_1m_t_end_idx,lasso_1m_z_start_idx:lasso_1m_z_end_idx],0),lasso_z[lasso_1m_z_start_idx:lasso_1m_z_end_idx],label='LASSO 1m')
    plt.plot(np.mean(clubb_wp2[silhs_t_start_idx:silhs_t_end_idx,silhs_z_start_idx:silhs_z_end_idx],0),silhs_z[silhs_z_start_idx:silhs_z_end_idx],label='CLUBB-SILHS')
    #plt.imshow(np.mean(lasso_w,(2,3)).T,origin='lower')
    plt.xlabel('[m2/s2]')
    plt.ylabel('height [m]')
    plt.title('LASSO, lidar, CLUBB wp2')
    plt.legend()
    #plt.subplots_adjust(bottom=0.2)
    plt.savefig(silhs_files[case]+'_wp2.pdf',dpi=300,bbox_inches="tight")
    plt.cla()
    
    plt.plot(np.nanmean(dl_wskew[dl_t_start_idx:dl_t_end_idx,dl_z_start_idx:dl_z_end_idx],0),dl_z[dl_z_start_idx:dl_z_end_idx],label='DL')
    plt.plot(np.mean(lasso_wskew[lasso_t_start_idx:lasso_t_end_idx,lasso_z_start_idx:lasso_z_end_idx],0),lasso_z[lasso_z_start_idx:lasso_z_end_idx],label='LASSO')
    plt.plot(np.mean(lasso_1m_wskew[lasso_1m_t_start_idx:lasso_1m_t_end_idx,lasso_1m_z_start_idx:lasso_1m_z_end_idx],0),lasso_z[lasso_1m_z_start_idx:lasso_1m_z_end_idx],label='LASSO 1m')
    plt.plot(np.mean(clubb_wskew[silhs_t_start_idx:silhs_t_end_idx,silhs_z_start_idx:silhs_z_end_idx],0),silhs_z[silhs_z_start_idx:silhs_z_end_idx],label='CLUBB-SILHS')
    #plt.imshow(np.mean(lasso_w,(2,3)).T,origin='lower')
    plt.xlabel('[-]')
    plt.ylabel('height [m]')
    plt.title('LASSO, lidar, CLUBB skewness')
    plt.legend()
    #plt.subplots_adjust(bottom=0.2)
    plt.savefig(silhs_files[case]+'_wp3.pdf',dpi=300,bbox_inches="tight")
    plt.cla()

