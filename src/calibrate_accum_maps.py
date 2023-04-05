#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:29:22 2023

@author: jason

code to apply regional bias functions to calibrate multi-annual accumulation maps

applicaiton of calibration would assume the samples from which the bias stats were made have no time-dependence.

accum=tp+E-BSS


have MARv3.13.0wBSS_15km grids

want NHM E and BSS grids if we make CARRA accum maps


"""


# import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import numpy as np
# from numpy.polynomial.polynomial import polyfit
import os
# from scipy import stats
# from datetime import datetime 
# os.environ['PROJ_LIB'] = r'C:/Users/Armin/Anaconda3/pkgs/proj4-5.2.0-ha925a31_1/Library/share' #Armin needed to not get an error with import basemap: see https://stackoverflow.com/questions/52295117/basemap-import-error-in-pycharm-keyerror-proj-lib
import geopandas as gp

# from matplotlib import gridspec

#-------------------------------------- change path
AD=0 #Armin
path='/Users/jason/Dropbox/CARRA/CARRA_rain/'

if AD:
    path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_rain/'
    path_csv='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA/'

os.chdir(path)


# ni=1269 ; nj=1069

# #-------------------------------- get masks
# fn=path+'ancil/2.5km_CARRA_west_lat_1269x1069.npy'
# lat=np.fromfile(fn, dtype=np.float32)
# lat=lat.reshape(ni, nj)
# # lat=np.rot90(lat.T)

# fn=path+'ancil/2.5km_CARRA_west_lon_1269x1069.npy'
# lon=np.fromfile(fn, dtype=np.float32)
# lon=lon.reshape(ni, nj)

# # ice mask
# fn2=path+'./ancil/CARRA_W_domain_ice_mask.nc'
# nc2 = xr.open_dataset(fn2)
# # print(nc2.variables)
# mask = nc2.z
# # mask=mask[::rf,::rf] #resampling
# # #mask = nc2.variables['z'][:,:]
# mask=np.asarray(mask)
# mask_svalbard=1;mask_iceland=1
# if mask_iceland:
#     mask[((lon-360>-30)&(lat<66.6))]=0
# if mask_svalbard:
#     mask[((lon-360>-20)&(lat>70))]=0

#%% bias statistics

bias_stats=pd.read_csv('/Users/jason/Dropbox/CARRA/CARRA_rain/stats/summary_vs_elev.csv')
# print(df)
print(bias_stats.columns)

#%% MAR
import xarray as xr
import datetime

# integrate SMB annually
def mass_integrate(var,varnam,nj,ni,mask,area):
    # if (( year%400 == 0) or (( year%4 == 0 ) and ( year%100 != 0))):
    #     print("%d is a Leap Year" %year)
    #     n_days=366
    # else:
    #     print("%d is Not the Leap Year" %year)
    #     n_days=365
    n_days=var.shape[0]
    var_cum=np.zeros((nj,ni))
    temp=np.zeros((nj,ni))

    daily=[]
    dates=[]
    
    # for dd in range(200,213):
    # for dd in range(19):
    for dd in range(n_days):
        # print(dd)
        date = datetime.date(year, 1, 1) ; delta = datetime.timedelta(dd) ; datex = date + delta ; date = pd.to_datetime(datex)                

        # var=np.rot90(var.T)
     
        # temp+=SMB[dd,0,:,:]
        result=var[dd,:,:]*mask
        temp+=result
        result2=np.sum(result*area*1e6)/1000/1e9
        daily.append(result2)
        # print(str(date.strftime('%Y %b %d')),result2)
        var_cum+=temp
        dates.append(str(date.strftime('%Y-%m-%d')))
        plot=0
        if plot:
            # plt.imshow(var_cum[dd,:,:])
            plt.imshow(result)
            plt.title('MAR '+varnam+' '+str(date.strftime('%Y %b %d')))
            plt.colorbar()
            plt.show()

    return(var_cum,n_days,daily,dates)

iyear=1991;fyear=1991

MAR_path='/Users/jason/0_dat/MAR_1991-2022/'
d = xr.open_dataset(MAR_path+'MARv3.13.0-15km-daily-ERA5-1997.nc')
mask=np.array(d.MSK[:,:])
# print(mask.shape)
AREA=np.array(d.AREA)
elev=np.array(d.SH)
plt.imshow(elev)


for year in range(iyear,fyear+1):
# for year in range(2012,2013):
    print(year)
    d = xr.open_dataset(MAR_path+'MARv3.13.0-15km-daily-ERA5-'+str(year)+'.nc')
    # RU=np.array(d.RU[:,0,:,:])
    SMB=np.array(d.SMB[:,0,:,:])
    n_days0=SMB.shape[0]
    nj=SMB.shape[1];ni=SMB.shape[2]

    # print(RU.shape)
    SMB_cum,n_days,daily,dates=mass_integrate(SMB,'SMB',nj,ni,mask,AREA)

#%%
plot=1
if plot:
    plt.close()
    # plt.imshow(ice_region)
    # plt.colorbar()
    plt.title('MAR x')
    plt.imshow(SMB_cum)
    plt.colorbar()
    plt.show()
#%% regional masks
fn="./ancil/mouginot_sectors/Greenland_Basins_PS_v1.4.2.shp"
# buffered shp to include the border region datapoints
# need to first create the shp below using GrIS_basemap_insitu_locations.py
fn="./ancil/mouginot_sectors/buffer_basins_include_border_regions.shp"

basins=gpd.read_file(fn)
basins=basins.to_crs('epsg:4326')  #change coord system
if basin_region!='':  #choose subregion
    # basins=basins[basins.SUBREGION1.isin(basin_region)]
    basins=basins[basins.SUBREGION1==basin_region]
# #create geodataframe
# df_scat_geo=gp.GeoDataFrame(df_scat, geometry= gp.points_from_xy(df_scat.lon, df_scat.lat))
# #search for matching points in basins
# df_basins = gp.sjoin(basins, gp.GeoDataFrame(df_scat_geo.geometry, crs='epsg:4326'), op='contains')
# df_basins=df_basins.sort_values(by='index_right')
# #compare indexes and drop values 
# df_scat = df_scat.loc[df_scat.index.isin(df_basins.index_right)]