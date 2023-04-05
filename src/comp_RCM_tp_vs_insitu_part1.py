#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 09:44:27 2021

@author: jeb and Armin

THIS CODE CAN TAKE ~60 sec to run for one graphic. AD? I think the slowness is because of grid reprojection?

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
# from scipy.interpolate import griddata
import pandas as pd
import geopandas as gp
import numpy as np
from numpy.polynomial.polynomial import polyfit
from shapely.geometry import Point, LineString
import os
from scipy import stats
from datetime import datetime 
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import datetime
from datetime import datetime
os.environ['PROJ_LIB'] = r'C:/Users/Armin/Anaconda3/pkgs/proj4-5.2.0-ha925a31_1/Library/share' #Armin needed to not get an error with import basemap: see https://stackoverflow.com/questions/52295117/basemap-import-error-in-pycharm-keyerror-proj-lib
# import xarray as xr
import time


#-------------------------------------- change path
AD=0 #Armin
path='/Users/jason/Dropbox/CARRA/CARRA_precip/'
path_point_data_RCMs_in_situ='/Users/jason/0_dat/RCMs_precip_eval/'

if AD:
    path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_rain/'
    path_point_data_RCMs_in_situ='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA/'

os.chdir(path)

stats_path='./stats/'
 
# GIS stuff
fn='./ancil/coastline/GRL_adm0.shp'

#-------------------------------------- start date as function of latitude
# realistic
monthSouth=10 ; daySouth=15
monthNorth=9 ; dayNorth=1

# # unrealistic
# monthSouth=9 ; daySouth=1
# monthNorth=8 ; dayNorth=15

day_of_year0 = datetime(2000,monthSouth,monthSouth).strftime('%j')
day_of_year1 = datetime(2000,monthSouth,daySouth).strftime('%j')
print(day_of_year0,day_of_year1)
y1=float(day_of_year1) ; y0=float(day_of_year0)
x0=60 ; x1=80
a1x=(y1-y0)/(x1-x0) ; a0=y1-(x1*a1x)
x=79

#---------------------------------------- user defined parameters
ly='p' # to png, set to 'x' for console
comp_by_elevation=1
elev_above_thresh=0
map_ratio=0 # can we use the ablation area snow pits. are we losing mass out the bottom of the snow profile AD?)

plt_name=1
plt_fit=1
pits=1
no_ACT1011=0
drop_dome_GP=1
ablation_only=0 ; suffixx='_ablation_only'
PROMICE_only=0 ; suffixx='_PROMICE_only'
ACT_only=0 ; suffixx='_ACT_only'
accumulation_only=0; suffixx='_accumulation_only'
regional_scatter=1
plt_legend=1

suffixx='_all'

hival=5000 #!!!!!!! careful to not exclude e.g. Q transect data!
# hival=1450
hivals=[hival]
            
#define start date method
accum_season_start_f_latitude=0
accum_season_start_original=0
accum_season_start_t2m=1

iyear=1997 # comparison starts this year

# ---------temperature thresholds used to identify the end of the previous melt season
t_thresh=-1.5

t_threshs=-(np.arange(0,1.5,0.05))+0.1
t_threshs=-(np.arange(0,3.2,0.1))+1.5
t_threshs=-(np.arange(0,2.,0.1))+0.75
t_threshs=[0]# now fixed at 0.0 which produced the most reasonable results!

# t_threshs=-(np.arange(0,0.1,0.05))+0.1
for t_thresh in t_threshs:
    # ---------------------------------- graphics setting1s
    fs=18 # font size
    fs=20 # font size
    ms=80 # marker size
    th=1
    # plt.rcParams['font.sans-serif'] = ['Georgia']
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.color'] = "grey"
    plt.rcParams["font.size"] = fs
    plt.rcParams['ytick.labelsize'] = fs  
    plt.rcParams['xtick.labelsize'] = fs
    plt.rc('legend',fontsize=fs) 
    
    # RCM_names=['MARv3.11.5_6km','CARRA']
    # RCM_names=['CARRA','RACMO2.3p2_5.5km','MARv3.11.5_6km']
    # RCM_names=['RACMO2.3p2_5.5km'] ; n_compvars=3
    # RCM_names=['MARv3.11.5_6km']; n_compvars=2
    # RCM_names=['RACMO2.3p2_5.5km','CARRA','NHM-SMAP','MARv3.11.5_6km']
    # RCM_names=['MARv3.11.5_6km','CARRA','RACMO2.3p2_5.5km']
    # RCM_names=['MARv3.11.5_10km','MARv3.11.5_15km','MARv3.11.5_20km']
    ## RCM_names=['MARv3.11.5_10km','MARv3.11.5_15km','MARv3.11.5_20km']    

    RCM_names=['CARRA','RACMO2.3p2_5.5km','NHM-SMAP']
    RCM_names=['CARRA','RACMO2.3p2_5.5km','NHM-SMAP','MARv3.11.5_6km','MARv3.13.0wBSS_15km','MARv3.13.0noBSS_15km']
    RCM_names=['CARRA','RACMO2.3p2_5.5km','NHM-SMAP','MARv3.13.0wBSS_15km','MARv3.13.0noBSS_15km']
    RCM_names=['CARRA','RACMO2.3p2_5.5km','NHM-SMAP','MARv3.13.0wBSS_15km','MARv3.13.0noBSS_15km']
    # RCM_names=['CARRA','RACMO2.3p2_5.5km','NHM-SMAP','MARv3.13.0wBSS_15km']
    # RCM_names=['CARRA','RACMO2.3p2_5.5km','NHM-SMAP','MARv3.11.5_6km','MARv3.13.0wBSS_15km','MARv3.13.0noBSS_15km']
    # RCM_names=['MARv3.12.0.4BS_15km']    
    # RCM_names=['MARv3.11.5_6km'] # no BSS    
    RCM_names=['MARv3.13.0wBSS_15km','MARv3.13.0noBSS_15km']
    # RCM_names=['MARv3.13.0noBSS_15km'] # BSS
    # RCM_names=['CARRA']
    # RCM_names=['RACMO2.3p2_5.5km']
    # RCM_names=['NHM-SMAP']
    comp_var='P-E-BSS'
    # comp_var='P-E'
    comp_var='P'

    # out_fn_all=RCM_names[0]+'_'+comp_var+'comp_vs_elev'+str(comp_by_elevation)+suffixx
    # out_fn_all=RCM_names[0]+'_'+comp_var+'comp_vs_elev'+str(comp_by_elevation)
    # out_fn_all=open(stats_path+out_fn_all+'.csv','w')
    # out_fn_all.write("RCM,comp_var,hival,slope,intercept,R,MAE: RCM - in-situ,RMSD anoms,mean obs,RMSD anoms/mean obs,|MAE|/mean obs,N,T_threshold\n")
    for jj,RCM_name in enumerate(RCM_names):
        n_compvars=3
        if jj>2:n_compvars=2
        for kk in range(0,1): # 'P'
        # for kk in range(1,2): # 'P-E'
        # for kk in range(2,3): # 'P-E-BSS'
            # comp_vars=['P-BSS']
            # comp_vars=['P-E','P']
            if kk==0:comp_var='P'
            if kk==1:comp_var='P-E'
            if kk==2:comp_var='P-E-BSS'
            # comp_vars=['P','P-E','P-E-BSS']
            # comp_vars=['P-E','P-E-BSS']
            
            years=np.arange(iyear,2021).astype('str')
            
            df_t2m = pd.DataFrame()
            df_tp = pd.DataFrame()
            df_E = pd.DataFrame()
            df_C = pd.DataFrame()
            df_BSS = pd.DataFrame()
            df_subl = pd.DataFrame()
            df_sndiv = pd.DataFrame()
            
            for year in years:
                # print('catting',RCM_name,year)
                fn=path_point_data_RCMs_in_situ+'_common_format/'+RCM_name+'_t2m_at_points_'+year+'.csv'
                dd=pd.read_csv(fn)
                print('catting',RCM_name,year)
                df_t2m = pd.concat([df_t2m,dd])
                # print(os.system('ls -lF '+fn))
                
                if RCM_name[0:3]=='MAR':
                    if RCM_name[0:8]=='MARv3.13':
                    #     fn=path_point_data_RCMs_in_situ+'_common_format/'+RCM_name+'_SMB_at_points_'+year+'.csv'
                    # else:
                        fn=path_point_data_RCMs_in_situ+'_common_format/'+RCM_name+'_SF_at_points_'+year+'.csv'
                        dd=pd.read_csv(fn)
                        df_tp = pd.concat([df_tp,dd])
                # print(RCM_name[0:8])
                #     #%%
                if RCM_name[0:3]=='NHM':
                    fn=path_point_data_RCMs_in_situ+'_common_format/'+'NHM-SMAP'+'_swgf_at_points_'+year+'.csv'
                    dd=pd.read_csv(fn)
                    df_E = pd.concat([df_E,dd])
                    fn=path_point_data_RCMs_in_situ+'_common_format/'+'NHM-SMAP'+'_BSS_at_points_'+year+'.csv'
                    dd=pd.read_csv((fn))
                    df_BSS = pd.concat([df_BSS,dd])
                    fn=path_point_data_RCMs_in_situ+'_common_format/'+RCM_name+'_tp_at_points_'+year+'.csv'
                    dd=pd.read_csv(fn)
                    df_tp = pd.concat([df_tp,dd])
                    # print('catting',fn)
                    
                if RCM_name[0:3]=='CAR':
                    fn=path_point_data_RCMs_in_situ+'_common_format/'+'NHM-SMAP'+'_swgf_at_points_'+year+'.csv'
                    dd=pd.read_csv(fn)
                    df_E = pd.concat([df_E,dd])
                    fn=path_point_data_RCMs_in_situ+'_common_format/'+'NHM-SMAP'+'_BSS_at_points_'+year+'.csv'
                    dd=pd.read_csv((fn))
                    df_BSS = pd.concat([df_BSS,dd])
                    fn=path_point_data_RCMs_in_situ+'_common_format/'+RCM_name+'_tp_at_points_'+year+'.csv'
                    # print('catting',fn)
                    # fn=path_point_data_RCMs_in_situ+'_common_format/'+'NHM-SMAP'+'_tp_at_points_'+year+'.csv'
                    dd=pd.read_csv(fn)
                    df_tp = pd.concat([df_tp,dd])

                if RCM_name[0:3]=='RAC':
                    fn=path_point_data_RCMs_in_situ+'_common_format/'+RCM_name+'_subl_at_points_'+year+'.csv'
                    dd=pd.read_csv(fn)
                    df_subl = pd.concat([df_subl,dd])
                    fn=path_point_data_RCMs_in_situ+'_common_format/'+RCM_name+'_sndiv_at_points_'+year+'.csv'
                    dd=pd.read_csv(fn)
                    df_sndiv = pd.concat([df_sndiv,dd])
                    fn=path_point_data_RCMs_in_situ+'_common_format/'+RCM_name+'_tp_at_points_'+year+'.csv'
                    dd=pd.read_csv(fn)
                    df_tp = pd.concat([df_tp,dd])
            
            # if comp_var=='P-E':
            #     PmE=df_tp.iloc[:,2:]+df_E.iloc[:,1:]

            if comp_var=='P-E-BSS':
                if RCM_name[0:3]=='CAR':
                    df_RCM=df_tp.iloc[:,1:]+df_E.iloc[:,2:]-df_BSS.iloc[:,2:]
                if RCM_name[0:3]=='NHM':
                    df_RCM=df_tp.iloc[:,2:]+df_E.iloc[:,2:]-df_BSS.iloc[:,2:]
                if RCM_name[0:3]=='MAR':
                    df_RCM=df_tp.iloc[:,1:]
                if RCM_name[0:3]=='RAC':
                    df_RCM=df_tp.iloc[:,2:]+df_subl.iloc[:,2:]-df_sndiv.iloc[:,2:]
          
            if comp_var=='P':
                if RCM_name[0:3]=='CAR':
                    df_RCM=df_tp.iloc[:,1:]
                if RCM_name[0:3]=='NHM':
                    df_RCM=df_tp.iloc[:,2:]
                if RCM_name[0:3]=='MAR':
                    df_RCM=df_tp.iloc[:,1:]
                if RCM_name[0:3]=='RAC':
                    df_RCM=df_tp.iloc[:,2:]
    
            # if comp_var=='P-E':
            #     df_RCM=PmE
            #     plt.plot(df_tp.iloc[:,1])
            #     df_RCM['time']= pd.to_datetime(df_tp.iloc[:,0], format='%Y-%m-%d')
            #     df_RCM = df_RCM.set_index(['time'])

            # df_RCM=PmEmS.copy()

            df_RCM['time']= pd.to_datetime(df_tp.time, format='%Y-%m-%d')
            df_RCM = df_RCM.set_index(['time'])
            
            df_t2m['time']= pd.to_datetime(df_t2m.time, format='%Y-%m-%d')
            df_t2m['time2'] = df_t2m.time
            df_t2m = df_t2m.set_index(['time'])
            
        #     #%%
        # # a=df_RCM.copy()
        # plt.plot(df_RCM['ACT-10a'])
        # plt.title(RCM_name)
        # #%%
            #-------------------------------------- read snow pit/core data
            fn='./ancil/Greenland_snow_pit_SWE_v20210626.xlsx'
            # os.system('open '+fn)
            df_insitu = pd.read_excel(fn, engine='openpyxl')

            df_insitu = df_insitu.drop(df_insitu[df_insitu["End Year"] <iyear].index)
            
            df_insitu = df_insitu.drop(df_insitu[df_insitu.Name=='Basin5'].index) # !!!
            df_insitu = df_insitu.drop(df_insitu[df_insitu.Name=='JAR2'].index) # !!!

            if drop_dome_GP:
                df_insitu = df_insitu.drop(df_insitu[df_insitu.Name=='Dome Gp'].index)

            if no_ACT1011:
                df_insitu = df_insitu.drop(df_insitu[df_insitu.Name=='ACT-10a'].index)
                df_insitu = df_insitu.drop(df_insitu[df_insitu.Name=='ACT-10b'].index)
                df_insitu = df_insitu.drop(df_insitu[df_insitu.Name=='ACT-10c'].index)
                df_insitu = df_insitu.drop(df_insitu[df_insitu.Name=='ACT-11b'].index)
                df_insitu = df_insitu.drop(df_insitu[df_insitu.Name=='ACT-11c'].index)
    
            if ablation_only:
                df_insitu = df_insitu.drop(df_insitu[df_insitu.Regime!='ablation'].index)
            if accumulation_only:
                df_insitu = df_insitu.drop(df_insitu[df_insitu.Regime!='accumulation'].index)
            if PROMICE_only:
                df_insitu = df_insitu.drop(df_insitu[df_insitu.Program!='PROMICE'].index)
            if ACT_only:
                df_insitu = df_insitu.drop(df_insitu[df_insitu.Source!='Forster/Box/McConnell'].index)
                df_insitu = df_insitu.drop(df_insitu[df_insitu.lon<-45].index)
    
                # df_insitu = df_insitu.drop(df_insitu[df_insitu.Name=='ACT-10b'].index)
                # df_insitu = df_insitu.drop(df_insitu[df_insitu.Name=='ACT-10c'].index)
                # df_insitu = df_insitu.drop(df_insitu[df_insitu.Name=='ACT-11b'].index)
                # df_insitu = df_insitu.drop(df_insitu[df_insitu.Name=='ACT-11c'].index)        
            df_insitu = df_insitu.reset_index(drop=True)

            data1=np.zeros((df_insitu.shape[0],8))*np.nan
            n = df_insitu.shape[0]
            # df_insitu.Name
        
            #-------------------------------------- start date
            if accum_season_start_t2m:
                #------- lat value for years where t2m always < some threshold
                df_insitu['doy']=a1x*df_insitu.lat+a0
                df_insitu['doy'][~np.isfinite(df_insitu['doy'])]=250
                df_insitu.doy=df_insitu.doy.round(0).astype(int)
                df_insitu.doy=df_insitu.doy.astype(str)    
                df_insitu['date00']= pd.to_datetime(df_insitu["Start Year"].astype(str)+' '+df_insitu.doy, format='%Y %j')
                
                #------ t2m method
                df_insitu['date0']=0
                yra =  pd.DatetimeIndex(df_t2m.time2).year
                for i in range(0,n):
                    name = df_insitu.Name[i]
                    yrb = df_insitu["Start Year"][i]
                    ff = 0
                    for j in range(1,len(df_t2m)): 
                        if (yra[j]==yrb) and (df_t2m[name][j]<=t_thresh) and (df_t2m[name][j-1]>t_thresh):
                            # print(j)
                            last_day = df_t2m.time2[j]
                            # new code for last_day that makes a datetime value that later causes a fail because it has an index, should be a timestamp
                            # last_day = pd.to_datetime({'year': [df_insitu['End Year'][i]],
                            #                            'month': [df_insitu['End Month'][i]],
                            #                            'day': [df_insitu['End Day'][i]]})
                            # last_day.date

                            ff = 1
                            df_insitu.date0[i] = pd.to_datetime(last_day, format='%Y-%m-%d %H:%M:%S')
                            #!!! using Q transect data to determine why 2019 values not making it into the comparison
                            # if name[0]=='Q':print(name,df_insitu['End Day'][i], df_insitu['End Month'][i], df_insitu['End Year'][i],df_insitu.date0[i],last_day)
                    if ff!=1:
                        df_insitu.date0[i] = df_insitu.date00[i]
                        # print(name,'no date found',df_insitu.date0[i]) #!!!
                #-------------------------------------- end date
                df_insitu['date1']= pd.to_datetime(df_insitu.TimeEnd, format='%Y-%m-%d %H:%M:%S')
                # df_insitu['year1']=pd.DatetimeIndex(df_insitu['date1']).year
                # df_insitu['mon1']=pd.DatetimeIndex(df_insitu['date1']).month
                # df_insitu['day1']=pd.DatetimeIndex(df_insitu['date1']).day
            
                # #%% End Year Histogram
                # hist=0
                # if hist: 
                #     data_core = df_insitu.loc[df_insitu['Type'] == "core"]
                #     data_pits = df_insitu.loc[df_insitu['Type'] == "pit"]
                #     data_pits = data_pits.reset_index()
                #     data_hist =pd.DataFrame()
                #     data_hist["core"] = data_core["End Year"]
                #     data_hist["pits"] = data_pits["End Year"]
                #     n1 = len(data_core)
                #     n2 = len(data_pits)
                #     years_bar=np.arange(1997,2020).astype('str')
                #     print(len(years_bar))
                
                #     fig4 = plt.figure()
                #     plt.hist(data_hist, bins=len(years_bar)-2, edgecolor="black", width=.4, align='left')
                #     plt.ylabel("quantity")
                #     labels= ["cores, N="+str(n1),"pits, N="+str(n2)]
                #     plt.legend(labels, fontsize=10, loc="upper center")
                
                #     if ly == 'x':
                #             plt.show()
                #     if ly =='p':
                #             fig_path='./Figs/tp_vs_insitu/'
                #             os.system('mkdir -p '+fig_path)
                #             # figname=fig_path+str(sum_tp)+datesq
                #             for DPI in DPIs:
                #                 figname=fig_path+'EndYear_histogram_pits_and_cores'
                #                 plt.savefig(figname+'.png', bbox_inches='tight', dpi=150)
                # df_insitu.Accumulation[df_insitu.Name=='Dome GP']
                # compare and make scatter
                #
                # loop over snow pit sites
                # for i in range(0,n):
                for i in range(0,n):
                    start = str(df_insitu.date0[i])  
                    # print(i,start,df_RCM.iloc[:,i])
                    end = str(df_insitu.date1[i])
                    name = df_insitu.Name[i]
                    accum = df_insitu.Accumulation[i]
                    RCM_tp_sum = df_RCM.loc[start:end, name].sum()
                    if ((df_insitu.date0[i].year>iyear)&(accum<hival)&(RCM_tp_sum<1e6)):
                        data1[i,0] = i
                        data1[i,1] = accum
                        data1[i,2] = RCM_tp_sum
                        data1[i,3] = df_insitu.lat[i]
                        data1[i,4] = df_insitu.lon[i]
                        data1[i,5] = RCM_tp_sum-df_insitu.Accumulation[i]
                        data1[i,6] = RCM_tp_sum-df_insitu.Accumulation[i]
                        data1[i,7] = df_insitu.Elevation[i]
    
            if regional_scatter: 
                basin_regions=['SE','NW','CE','CW', 'SW', 'NO', 'NE']
                basin_ranges=[2000, 600, 800, 550, 1600, 400, 140]
                # basin_regions=['SE']
                # basin_ranges=[1600]
            else:
                basin_regions=['all_basins']
                basin_ranges=[1200]

            # -------------------------------- loop over regional basins
            for basin_index,basin_region in enumerate(basin_regions):  
                
                plt.close()
                
                fig, ax = plt.subplots(figsize=(9.6,10))
    
                low=10 # by adjusting low and high, can evaluate different regions in the scatter
                # v=[(np.isfinite(x)&(np.isfinite(y))&(x>low)&(y>low))]
    
                interp_form='nearest neighbor'

                print(basin_region)
                
                out_fn=basin_region+'_'+RCM_names[jj]+'_'+comp_var+'comp_vs_elev'+str(comp_by_elevation)+'_t2m'+str("%.0f"%t_thresh)#+'_'+str(lower_elevation_threshold)

                out_f1=open(stats_path+out_fn+'.csv','w')
                out_f1.write("RCM,comp_var,slope,intercept,R,RCM - in-situ,bias adj. RMSD,mean obs,RMSD/mean obs,std. obs,N,min_elev,max_elev\n")
                #---------------------------------- new scatter
                df_scat = pd.DataFrame()
                df_scat["x"] = data1[:,2]
                df_scat["y"] = data1[:,1]
                df_scat["obs"] = data1[:,1]
                df_scat["model"] = data1[:,2]
        
                if comp_by_elevation:
                    df_scat["y"] = data1[:,2]-data1[:,1]
                    df_scat["rat"] = data1[:,2]/data1[:,1]
                    df_scat["rat"][data1[:,1]==0]=np.nan
                    df_scat["x"] = data1[:,7] 
                # data1.shape
                # df_insitu.shape
                df_scat["source"] = df_insitu.Program
                df_scat["source"] = df_insitu.Program
                df_scat["y1"] = df_insitu["End Year"]
                df_scat["y0"] = df_insitu["Start Year"]
                df_scat["name"] = df_insitu["Name"]
                df_scat["lat"] = df_insitu["lat"]
                df_scat["lon"] = df_insitu["lon"]
                df_scat["elevation"] = df_insitu["Elevation"]
                    
                df_scat = df_scat.drop(df_scat[~np.isfinite(df_scat.x)].index)
                df_scat = df_scat.drop(df_scat[~np.isfinite(df_scat.y)].index)
                # df_scat = df_scat.drop(df_scat[df_scat.source!='PROMICE'].index)
   
                if comp_by_elevation==0:
                    df_scat = df_scat.drop(df_scat[df_scat.x < low].index)
                    df_scat = df_scat.drop(df_scat[df_scat.y < low].index)

                if regional_scatter==1: 
                    #import basins shapefile -> original
                    fn="./ancil/mouginot_sectors/Greenland_Basins_PS_v1.4.2.shp"
                    # buffered shp to include the border region datapoints
                    # need to first create the shp below using GrIS_basemap_insitu_locations.py
                    fn="./ancil/mouginot_sectors/buffer_basins_include_border_regions.shp"
                    
                    basins=gpd.read_file(fn)
                    basins=basins.to_crs('epsg:4326')  #change coord system
                    if basin_region!='':  #choose subregion
                        # basins=basins[basins.SUBREGION1.isin(basin_region)]
                        basins=basins[basins.SUBREGION1==basin_region]
                    #create geodataframe
                    df_scat_geo=gp.GeoDataFrame(df_scat, geometry= gp.points_from_xy(df_scat.lon, df_scat.lat))
                    #search for matching points in basins
                    df_basins = gp.sjoin(basins, gp.GeoDataFrame(df_scat_geo.geometry, crs='epsg:4326'), op='contains')
                    df_basins=df_basins.sort_values(by='index_right')
                    #compare indexes and drop values 
                    df_scat = df_scat.loc[df_scat.index.isin(df_basins.index_right)]
                 
                    
                # df_scat = df_scat.drop(df_scat[df_scat.source == 'ACT-10/11'].index)
                df_scat = df_scat.reset_index(drop=True)
                
                #define plot parameters    
                mark =  ['o', 's','o', 'v',    'D',           '>', '<',      's','*', 'v','+']
                color = ['r', 'b', 'c', 'grey', 'darkviolet',  'b', 'darkorange','k','m', 'brown','orange']
                source = df_scat.source.unique()
                n_source=len(source)
                len(mark)
                mark=mark[0:n_source]
                color=color[0:n_source]
                pd_source = pd.DataFrame({'source': source, 'color': color, 'marker': mark})
                
                for i in range(len(df_scat)):
                    site1=df_scat.source[i]
                    ind = pd_source.source[pd_source.source == site1].index[0]
                    marker=str(pd_source.marker[ind])
                    col=str(pd_source.color[ind])
                    # print(col)
                    ax.scatter(df_scat.x[i], df_scat.y[i], facecolors='None',
                               edgecolors=col, marker=marker, s=ms, label=site1,
                               linewidth=2, linestyle = 'None')
            
                if RCM_name=='CARRA':
                    handles, labels = ax.get_legend_handles_labels() #To group labels and only display once
                    by_label = dict(zip(labels, handles))
                    if plt_legend:
                        leg_loc='lower left'
                        if basin_region=='SW':leg_loc='upper right'
                        ax.legend(by_label.values(), by_label.keys(), fontsize=fs*0.7,loc=leg_loc)
                        leg = ax.get_legend()   
                        if regional_scatter==0:
                            leg.set_bbox_to_anchor((1, .29))   #put legend next to map                       
            
                xx=[np.min(df_scat.x), np.max(df_scat.x)]
                
                y_all=df_scat.y 
                x_all=df_scat.x 
                
                MAE=np.nanmean(df_scat["model"]-df_scat["obs"])
                bias_adjusted_x=df_scat["model"]+MAE
                RMSDx=np.sqrt(np.nanmean((bias_adjusted_x-df_scat["obs"])**2))
                # RMSDx=np.sqrt(np.mean((bias_adjusted_x-df_scat["obs"])**2))
        
                if basin_region=='SW' and comp_by_elevation:
                    d_elev=300
                    elev_bins=np.arange(300,2100,d_elev)
                    elev_bin_bias=np.zeros(len(elev_bins))
                    temp=np.array(df_scat.x)
                    for elev_index,elev_bin in enumerate(elev_bins):
                        v=np.where((temp>=elev_bin)&(temp<=elev_bin+d_elev))
                        elev_bin_bias[elev_index]=np.nanmean(y_all[v[0]])
                        if len(v[0])==0:elev_bin_bias[elev_index]=np.nan
                    # plt.plot(elev_bins,elev_bin_bias,drawstyle='steps',linewidth=th*2,color='grey')
                    plt.step(elev_bins,elev_bin_bias,linewidth=th*2,color='grey',where='mid')
                
                N=len(df_scat)
                if basin_region!='SW':
                    b, a1 = polyfit(x_all,y_all, 1)
                else:
                    b, a1 = polyfit(x_all[x_all>1700],y_all[x_all>1700], 1)
                coefs=stats.pearsonr(x_all,y_all)
                lab=', R = '+str("%.2f"%coefs[0]+', MAE = '+str("%.2f"%MAE)+', N = '+str(N))
    
    
                if plt_fit:
                    xx=[np.min(x_all),np.max(x_all)]
                    if basin_region=='SW':
                        xx[0]=1700
                    if basin_region!='CE':
                        ax.plot([xx[0],xx[1]], [b + a1 * xx[0],b + a1 * xx[1]], '--',linewidth=th*4,c='grey')
            
                percent_bias=MAE/abs(np.mean(y_all))
                
                out_f1.write(RCM_name+
                              ','+comp_var+
                              # ','+str(hival)+
                                  ','+str("%.2f"%a1)+                             
                                  ','+str("%.0f"%b)+                             
                                  ','+str("%.3f"%coefs[0])+                             
                                  ','+str("%.0f"%MAE)+
                                  ','+str("%.0f"%RMSDx)+
                                  ','+str("%.0f"%(np.mean(df_scat["obs"])))+
                                  ','+str("%.2f"%(RMSDx/abs(np.mean(df_scat["obs"]))))+
                                  ','+str("%.0f"%(np.nanstd(df_scat["obs"])))+
                                  ','+str("%.0f"%N)+
                                  # ','+str("%.2f"%t_thresh)+
                                  ','+str("%.0f"%(np.nanmin(x_all)))+
                                  ','+str("%.0f"%(np.nanmax(x_all)))+
                                  '\n')
    
                rat=np.mean(y_all)/np.mean(x_all)
                # biases.append(percent_bias)
    
                props = dict(boxstyle='round', facecolor='w', alpha=0.6,edgecolor='w')
                
                mult=0.7
                xx0=0.02 ; yy0=0.98
    
                confidencex=1-coefs[1]
                
                add=[]
                if regional_scatter: add='\nbasin: '+basin_region
                # if elev_above_thresh: add='\nelevation: >'+str(elev_thresh)
                # if regional_scatter==1 and elev_above_thresh==1: add='\nbasin: '+basin_region+', elevation above '+str(elev_thresh)+' m'
                
                if comp_by_elevation==0:
                    basin_names=str(basin_region).replace('[','')
                    basin_names=basin_names.replace(']','')
                    basin_names=basin_names.replace("'","")
                    msg='in-situ = %.2f'%a1 +' * '+name_x+' + %.0f'%b +\
                        ',' +' R: %.3f'%coefs[0]+', ' + 'N: %.f'%N+'\n' +\
                            'model÷in-situ'+': %.2f'%rat+', model - in-situ'+': '+str("%.0f"%MAE)+' mm'\
                                '\nRMSD/mean:'+str("%.2f"%(RMSDx/abs(np.mean(y_all))))+', difference/mean:'+\
                                    ' '+str("%.2f"%((MAE)/abs(np.mean(y_all))))+add
                else:
                    # msg=RCM_name+' '+comp_var+'\nbasin: '+str(basin_region)+', N: %.f'%N
                    name_x=RCM_name
                    if RCM_name=='CARRA':name_x='CARRA hybrid with NHM-SMAP'
                    msg=str(basin_region)+' sector\n'+name_x+'\n'+\
                            'N: %.f'%N+'\n'+\
                            '$\overline{model - in situ}$: '+str("%.0f"%np.nanmean(df_scat.y))+' mm or ' +\
                            str("%.0f"%(100*np.nanmean(df_scat.y)/np.nanmean(df_scat["obs"])))+'%'+'\n' +\
                            '$\overline{in situ}$: '+str("%.0f"%np.nanmean(df_scat["obs"]))+' mm\n' +\
                            'bias-adj. RMSD: '+str("%.0f"%RMSDx)+' mm'
                            # 'bias-adj. RMSD: '+str("%.0f"%RMSDx)+' mm\nR: %.3f'%coefs[0]+', ' + 'N: %.f'%N
                            # 'average model÷in-situ'+': %.2f'%np.nanmean(df_scat.rat)+', '
                        # 'difference / km = %.0f'%(a1*1000)  +\

                ax.text(xx0,yy0,msg
                    ,transform=ax.transAxes, 
                    fontsize=fs*0.8,
                    verticalalignment='top', bbox=props,rotation=0,color='k', rotation_mode="anchor")
        
                hi=hival
                lo=-hi
                if comp_by_elevation==0:
                    ax.plot([lo,hi], [lo,hi], '-',c='k')
                    ax.set_xlim(0,hi)
                    ax.set_ylim(0,hi)
                    ax.set_xlabel(RCM_name+' '+comp_var+', mm w.e.', fontsize=fs)
                    ax.set_ylabel('in-situ snow accumulation, mm w.e.',fontsize=fs)
        
                else:
                    # ax.set_xlim(np.min(x_all)*0.9,np.max(x_all)*1.1)
                    ax.set_xlim(0,np.max(x_all)*1.1)
                    ax.set_xlabel('elevation, m a.s.l.', fontsize=fs)
                    ax.plot([np.min(x_all)*0.9,np.max(x_all)*1.1], [0,0], '-',c='k')
                    
                    ax.set_ylim(-basin_ranges[basin_index],basin_ranges[basin_index])
                    ax.set_ylabel('modeled minus in-situ snow accumulation, mm w.e.',fontsize=fs)
        
                ax.tick_params(labelsize=fs,labelrotation = 'auto')
            
                # plt.legend(prop={'size': fs*mult})
                    
                if ly == 'x':
                    plt.show()
                                
                DPIs=[150,600]
                DPIs=[150]
                
                if ly =='p':
                    fig_path='./Figs/C_vs_insitu/'
                    os.system('mkdir -p '+fig_path)
                    # figname=fig_path+str(sum_tp)+dates
                    for DPI in DPIs:
                        figname=fig_path+out_fn
                    plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)
    
                # plt.plot(elevs,biases,linestyle='steps')
                # elev_stats = pd.DataFrame()
                # elev_stats["elev"]=elevs
                # elev_stats["ppt_bias"]=biases
                # elev_stats.to_csv('./stats/'+RCM_name+'_P-E-BSS_elevstats.csv')

                out_f1.close()
            
                # make_gif=0
                # if make_gif:
                #     # fig_path='/Users/jason/Dropbox/CARRA/CARRA_rain/Figs/tp_vs_insitu/'
                #     fig_path='/Users/jason/Dropbox/CARRA/CARRA_rain/Figs/C_vs_insitu/'
                #     # animpath='/Users/jason/Dropbox/CARRA/CARRA_rain/Figs/tp_vs_insitu/anim/'
                #     animpath='/Users/jason/Dropbox/CARRA/CARRA_rain/Figs/C_vs_insitu/anim/'
                #     # msg='convert -delay 50 -loop 0 '+fig_path+'NHM-SMAP_P-E-BSScomp_vs_elev0*'+suffixx+'.png '+animpath+'NHM-SMAP_P-E-BSScomp_vs_elev0_all_T_sensitivity.gif'
                #     msg='convert -delay 50 -loop 0 '+fig_path+basin_region+'_MARv3.12.0.4*.png '+animpath+basin_region+'_MARv3.12.0.4.gif'
                #     # os.system('ls ./Figs/tp_vs_insitu/ NHM-SMAP_P-E-BSScomp_vs_elev0*')
                #     print('making gif')
                #     os.system(msg)
                #     os.system('ls -lF '+fig_path+basin_region+'_MARv3.12.0.4*.png')
                #     print("done")