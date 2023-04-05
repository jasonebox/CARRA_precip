#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 08:33:35 2022

@author: jason
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

# from matplotlib import gridspec


#-------------------------------------- change path
AD=0 #Armin
path='/Users/jason/Dropbox/CARRA/CARRA_precip/'

if AD:
    path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_rain/'
    path_csv='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA/'

os.chdir(path)

stats_path='./stats/'

RCM_names=['MARv3.11.5_6km']    
# 'MARv3.12.0.4BS_15km',
RCM_names=['CARRA','MARv3.12.0.4BS_15km','NHM-SMAP','RACMO2.3p2_5.5km'] 
RCM_names=['CARRA','RACMO2.3p2_5.5km','NHM-SMAP','MARv3.13.0wBSS_15km','MARv3.13.0noBSS_15km']
# RCM_names=['CARRA','RACMO2.3p2_5.5km','NHM-SMAP','MARv3.13.0wBSS_15km']

comp_var='P-E-BSS'
# comp_var='P'

t_thresh=0

#---------------------------------------- user defined parameters
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

ly='x' # to png, set to 'x' for console
comp_by_elevation=1
elev_above_thresh=0

basin_regions=['SE','NW','CE','CW', 'SW', 'NO', 'NE']
basin_ranges=[2000, 600, 800, 1200, 1200, 400, 400]
# basin_regions=['CW']
# basin_ranges=[3000]

rcm_color=['g', 'b', 'm', 'orange']
label=['CARRA', 'MAR 6km', 'NHM-SMAP', 'RACMO 5.5km']

nrow = 2
ncol = 4

# fig = plt.figure(figsize=(12, 12))
# gs = fig.add_gridspec(nrow, ncol, hspace=0, wspace=0)
# (ax1,ax2,ax3), (ax4, ax5) = 

do_plot=0

if do_plot:
    fig = plt.figure(constrained_layout=True,figsize=(9, 12))
    from matplotlib.gridspec import GridSpec
    
    gs = GridSpec(ncol, nrow, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[2,0])
    ax4 = fig.add_subplot(gs[3,0])
    
    ax5 = fig.add_subplot(gs[0,1])
    ax6 = fig.add_subplot(gs[1,1])
    ax7 = fig.add_subplot(gs[2,1])
    ax8 = fig.add_subplot(gs[3,1])

# gs.subplots(sharex='col', sharey='row')
# fig.suptitle("GridSpec")
# format_axes(fig)

df_all=pd.DataFrame()

for basin_index,basin_region in enumerate(basin_regions):  

    # fig, ax = plt.subplots(figsize=(9.6,10))

    panel_x_counter=0
    panel_y_counter=0
    # if basin_region>1:
    #     panel_x_counter=1
    #     panel_y_counter=1
    
    y0=-500
    y1=-y0
    for jj,RCM_name in enumerate(RCM_names):
    
        out_fn=stats_path+basin_region+'_'+RCM_names[jj]+'_'+comp_var+'comp_vs_elev'+str(comp_by_elevation)+'_t2m'+str("%.0f"%t_thresh)+'.csv'
        # print(out_fn)        
        print(os.system('ls -lF '+out_fn))
        df=pd.read_csv(out_fn)
        df['region']=basin_region
        df_all=pd.concat([df_all,df])
        print(df.columns)
        
        if do_plot:
    
            elevs=[0,3200]
            # df.hival
            elevs=[df.min_elev.values,df.max_elev.values]
            x0=0 ; x1=3400
    
            if basin_region=='NO':
                ax1.plot(elevs,df.slope.values*elevs+df.intercept.values,color=rcm_color[jj],label=RCM_name)
                ax1.set_title(basin_region)        
                ax1.set_ylim(y0,y1)
                ax1.axes.xaxis.set_ticklabels([])
                ax1.set_xlim(x0,x1)
            if basin_region=='NW':
                ax2.plot(elevs,df.slope.values*elevs+df.intercept.values,color=rcm_color[jj],label=RCM_name)
                ax2.set_title(basin_region)
                ax2.set_ylim(y0,y1)
                ax2.axes.xaxis.set_ticklabels([])
                ax2.set_xlim(x0,x1)
            if basin_region=='CW':
                ax3.plot(elevs,df.slope.values*elevs+df.intercept.values,color=rcm_color[jj],label=RCM_name)
                ax3.set_title(basin_region)
                ax3.set_ylim(y0,y1)
                ax3.axes.xaxis.set_ticklabels([])
                ax3.set_xlim(x0,x1)
            if basin_region=='SW':
                ax4.plot(elevs,df.slope.values*elevs+df.intercept.values,color=rcm_color[jj],label=RCM_name)
                ax4.set_title(basin_region)
                ax4.set_ylim(y0,y1)
                # ax4.axes.xaxis.set_ticklabels([])
                ax4.set_xlim(x0,x1)
    
            if basin_region=='NE':
                ax5.plot(elevs,df.slope.values*elevs+df.intercept.values,color=rcm_color[jj],label=RCM_name)
                ax5.set_title(basin_region)        
                ax5.set_ylim(y0,y1)
                ax5.axes.xaxis.set_ticklabels([])
                ax5.set_xlim(x0,x1)
            if basin_region=='CE':
                ax6.plot(elevs,df.slope.values*elevs+df.intercept.values,color=rcm_color[jj],label=RCM_name)
                ax6.set_title(basin_region)
                ax6.set_ylim(y0,y1)
                ax6.axes.xaxis.set_ticklabels([])
                ax6.set_xlim(x0,x1)
            if basin_region=='SE':
                ax7.plot(elevs,df.slope.values*elevs+df.intercept.values,color=rcm_color[jj],label=RCM_name)
                ax7.set_title(basin_region)
                ax7.set_ylim(y0,y1)
                ax7.set_ylim(y0,y1)
                ax7.set_xlim(x0,x1)
    
                # ax7.axes.xaxis.set_ticklabels([])
            # if basin_region=='SW':
            #     ax4.plot(elevs,df.slope.values*elevs+df.intercept.values,color=rcm_color[jj],label=RCM_name)
            #     ax4.set_title(basin_region)
            #     ax4.set_ylim(y0,y1)
                # ax4.axes.xaxis.set_ticklabels([])
            # ax=plt.subplots
    
            xx0=0.1 ; yy0=0.6 ; dy2=-0.15
            mult=0.9
            plt.text(xx0, yy0+jj*dy2, RCM_name,
                    fontsize=fs*mult,color=rcm_color[jj],transform=ax8.transAxes) 
        # plt.tight_layout()
        # fig.suptitle('RCM P-E-BSS difference with obs')
        # fig.supylabel('modeled minus in-situ snow accumulation, mm w.e.')
        # fig.supxlabel('elevation, m')

##%%
    df_all.columns
    nams=['region','RCM', 'slope', 'intercept','RCM - in-situ', 'bias adj. RMSD', 'mean obs',
           'RMSD/mean obs', 'N',  'min_elev', 'max_elev']
    df_all.to_csv('./stats/'+comp_var+'_summary_vs_elev.csv',columns=nams,index=None)
    # plt.legend()
    # plt.ioff()

#%%

df=pd.read_csv('/Users/jason/Dropbox/CARRA/CARRA_rain/stats/summary_vs_elev.csv')
# print(df)
print(df.columns)

#%%
df['abs_slope']=abs(df.slope)
df['abs_bias']=abs(df['RCM - in-situ'])

basin_regions=['SE','NW','CE','CW', 'SW', 'NO', 'NE']

for basin_region in basin_regions:
    v=df.region==basin_region
    
    vmin_RMSD=np.where((df.region==basin_region)&(df['bias adj. RMSD']==np.min(df['bias adj. RMSD'][v])))
    min_rmsd=np.array(np.min(df['bias adj. RMSD'][v]))
    min_rmsd_name=np.array(df.RCM[vmin_RMSD[0]])[0]
    
    slopes=np.array(df['abs_slope'][v])
    slopes_names=np.array(df['RCM'][v])
    vmin_slope=np.where(slopes==np.min(slopes))
    min_slope=np.array(np.min(df['abs_slope'][v]))
    min_slope_name=slopes_names[slopes==min(slopes)][0]

    bias=np.array(df['abs_bias'][v])
    bias_names=np.array(df['RCM'][v])
    vmin_bias=np.where(bias==np.min(bias))
    min_bias=np.array(np.min(df['abs_bias'][v]))
    min_bias_name=bias_names[bias==min(bias)][0]
    
    # print(basin_region,'min RMSE',min_rmsd_name,min_rmsd)
    print(basin_region,'min slope',min_slope_name,min_slope)

    # print(basin_region,'min bias',min_bias_name,min_bias,'min RMSE',min_rmsd_name,min_rmsd,'min slope',min_slope_name,min_slope)
    # print(basin_region,'min bias',min_bias_name,min_bias,'min RMSE',min_rmsd_name,min_rmsd,'min slope',min_slope_name,min_slope)
#%%
from pyproj import Proj, transform
import geopandas as gpd

inProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:3413')

ly='x'
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

plt_coastline=0

if plt_coastline:
    # workaround https://gis.stackexchange.com/questions/114066/handling-kml-csv-with-geopandas-drivererror-unsupported-driver-ucsv/346084
    fn='./ancil/coastline/GRL_adm0.shp'
    coastline = gpd.read_file(fn)
    coastline.crs = {'init' :'epsg:4326'}
    coastline = coastline.to_crs({'init': 'EPSG:3413'})
    # print(coastline.columns)

iyear=1997

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

plt.close()
fig, (ax, ax2) = plt.subplots(1, 2,figsize=(15,10), gridspec_kw={'width_ratios': [.75, .5]})

# -----------------------------------------------------------------------------
x0x=-6.6e5 ; y0x=-3.4e6
x1x=8.7e5 ; y1x=1221304.695
ax2.set_xlim([x0x,x1x])
# ax2.set_ylim([y0,y1])
ax2.axis('off')


df_insitu['lon_3413']=np.nan
df_insitu['lat_3413']=np.nan

# loop over sites to transform lat and lon
for i in range(len(df_insitu)): 
# for i in range(2): # for testing
    lat=df_insitu.lat[i]
    lon=df_insitu.lon[i]
    if lat>0: 
        df_insitu['lon_3413'][i], df_insitu['lat_3413'][i] = transform(inProj,outProj,lon,lat)
        # if plt_name:
        #     plt.text(lon_3413+1, lat_3413,df_insitu.Name[data1[i,0]],color='w', fontsize=fs*mult*1.2,verticalalignment='center')
        #     plt.text(lon_3413+1, lat_3413,df_insitu.Name[data1[i,0]],color='k', fontsize=fs*mult,verticalalignment='center')
    
#geodataframe to colorize bias
# data_3413.shape
# data_3413[:,6]
# print("average difference",np.nanmean(data_3413[:,6]))
if plt_coastline:
    coastline.plot(facecolor='lightgrey',
                edgecolor='k',linewidth=0.5,ax=ax2,figure=fig)
ylims=ax2.get_ylim()
xlims=ax2.get_xlim()
# # %%
# get list of site names
temp=df_scat.drop_duplicates(subset='name', keep="last")
site_names=temp.name.values

# populate arrays of average by site
mean_x=[] ; mean_y=[] ; N_in_mean=[] ; lon_3413=[] ; lat_3413=[]

for i,site in enumerate(site_names):

    tx, ty = transform(inProj,outProj,np.array(df_scat.lon[df_scat.name==site])[0],np.array(df_scat.lat[df_scat.name==site])[0])
    lon_3413.append(tx)
    lat_3413.append(ty)
    mean_y.append(np.mean(df_scat.y[df_scat.name==site]))
    mean_x.append(np.mean(df_scat.x[df_scat.name==site]))
    N_in_mean.append(np.sum([df_scat.name==site]))
    print(site,mean_x[i],mean_y[i],N_in_mean[i])

diff=np.array(mean_x)-np.array(mean_y)
    
gdf = gpd.GeoDataFrame(diff, geometry=gpd.points_from_xy(np.array(lon_3413), np.array(lat_3413)))
gdf.columns = ['difference','geometry']
        
# hix=500
# # create the colorbar
# mpl.rcParams['ytick.labelsize'] = fs  #To adjust tick fontsize everywhere
# if map_ratio:
#     norm = colors.Normalize(vmin_RMSD=-gdf.diff.max()+2, vmax=gdf.ratio.max())
# else:
#     norm = colors.Normalize(vmin_RMSD=-hix, vmax=hix)
    
# cbar = plt.cm.ScalarMappable(norm=norm, cmap='bwr')

# #divider = make_axes_locatable(ax2)
# #cax = divider.append_axes("right", size="5%", pad=0.1)

# #plot gdf
# gdf.plot(ax=ax2, figure = fig, markersize=ms,  column = 'difference', legend=False,
#           cmap='bwr', marker='o', edgecolor='k', linewidth=0.3,vmin_RMSD=-hix, vmax=hix)
# cb_ax = fig.axes[1] 
# cb_ax.tick_params(labelsize=fs) #to adjust tick fontsize


    
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


    # if ly =='p':
    #     fig_path='./Figs/C_vs_insitu/'
    #     os.system('mkdir -p '+fig_path)
    #     # figname=fig_path+str(sum_tp)+dates
    #     for DPI in DPIs:
    #         figname=fig_path+out_fn
    #         plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)
