#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:35:29 2021

@author: jeb
"""

from mpl_toolkits.basemap import Basemap

import geopandas as gp
import geopandas as gpd
import xarray as xr
import matplotlib as mpl
import pandas as pd

import numpy as np
# from numpy.polynomial.polynomial import polyfit
import os
import matplotlib.pyplot as plt


#-------------------------------------- change path
AD=0 #Armin
path='/Users/jason/Dropbox/CARRA/CARRA_rain/'

if AD:
    path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_rain/'
    path_csv='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA/'

os.chdir(path)

ly='p'

iyear=1997

no_ACT1011=0
drop_dome_GP=1
ablation_only=0 ; suffixx='_ablation_only'
PROMICE_only=0 ; suffixx='_PROMICE_only'
ACT_only=0 ; suffixx='_ACT_only'
accumulation_only=0; suffixx='_accumulation_only'

#import basins shapefile -> original
fn="./ancil/mouginot_sectors/Greenland_Basins_PS_v1.4.2.shp"
# buffered shp to include the border region datapoints
#@ Jason you need to first create the shp below using GrIS_basemap_insitu_locations.py

basin_regions=['SE','NW','CE','CW', 'SW', 'NO', 'NE']


basemap=0
res='l'
if basemap:
    ly='p'
    res='l'
if ly=='p':res='h'

ni=1269 ; nj=1069
subset_it=1
#---------------------------------------------------------- read in data 
fn='./ancil/CARRA_W_domain_elev.nc'
ds=xr.open_dataset(fn)
elev=ds.z
# print(elev.shape)
# elev[elev < 1] = np.nan

# read ice mask
fn2='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = xr.open_dataset(fn2)
# print(nc2.variables)
mask = nc2.variables['z'][:,:]
#mask = np.rot90(mask.T)
#plt.imshow(mask)

fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)



th=1

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

gdf = df_insitu
gdf = gdf.drop(gdf[gdf.Program == "Kjær et al, unpublished 2021"].index)
gdf = gdf.reset_index(drop=True)

#define plot parameters    
m =     ['o', 'D',          's','o', '*', 'D', 'D', '+', 'x', 'o','<']
color = ['r', 'darkviolet', 'k','grey', 'grey', 'y', 'r', 'k', 'y', 'b','c']
source = gdf.Program.unique()
pd_source = pd.DataFrame({'source': source, 'color': color[0:len(source)], 'marker': m[0:len(source)]})
    
subset_name=''; lonrot=0

max_value=12
max_value=36

elev0=50;elev1=3000;delev=500

if subset_it:
    elev0=0;elev1=3500;delev=500
    # 50,3000,200
    max_value=24
    max_value=36
    subset_name='subset_'
    lon=np.rot90(lon.T)
    lat=np.rot90(lat.T)
    xc0=100 ; xc1=810
    yc0=130 ; yc1=1150

    lat=lat[yc0:yc1,xc0:xc1]
    lon=lon[yc0:yc1,xc0:xc1]
    #print(lat.shape)
    ni=lat.shape[0]
    nj=lat.shape[1]
    lon=np.rot90(lon.T)
    lat=np.rot90(lat.T)
    
    elev=np.rot90(elev.T)
    elev=elev[yc0:yc1,xc0:xc1]
    elev=np.rot90(elev.T)
    
    mask=np.rot90(mask.T)
    mask=mask[yc0:yc1,xc0:xc1]
    mask=np.rot90(mask.T)
    #lonrot=11.25

    LLlat=lat[0,0]
    LLlon=lon[0,0]-360
    lon0=lon[int(round(ni/2)),int(round(nj/2))]-360
    lat0=lat[int(round(ni/2)),int(round(nj/2))]
    URlat=lat[ni-1,nj-1]
    URlon=lon[ni-1,nj-1]
    
    m = Basemap(llcrnrlon=LLlon, llcrnrlat=LLlat, urcrnrlon=URlon, urcrnrlat=URlat,\
                lat_0=lat0, lon_0=lon0+lonrot, \
                resolution=res, projection='lcc')
    
    x, y = m(lon, lat)


# --------------------------------------------------------- set up figure
# ax,fig = plt.figure(1, figsize=(8, 14), dpi=150) #frameon=False
fig, ax = plt.subplots(figsize=(9,15))
# fig.add_axes()
# ax = plt.gca()
    
elev[mask<=0]=np.nan  #mask elevation values for ice values only 
clevs=np.arange(elev0,elev1,delev) #define min, max and step 
con = m.contourf(x,y,elev, clevs, linewidths=th/2, cmap='Blues_r', alpha=.85)
m.drawcoastlines()



fn="./ancil/mouginot_sectors/buffer_basins_include_border_regions.shp"
basins=gpd.read_file(fn)
basins=basins.to_crs('epsg:4326')  #change coord system
    
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

gdf = df_insitu
gdf = gdf.drop(gdf[gdf.Program == "Kjær et al, unpublished 2021"].index)
gdf = gdf.reset_index(drop=True)

for i in range(len(gdf)): 
    if gdf.lat[i]>0: 
        lon1, lat1 = m(*(gdf.lon[i], gdf.lat[i])) #transform into m map projection
        val=gdf.Program[i]
        ind = pd_source.source[pd_source.source == val].index[0]
        marker=str(pd_source.marker[ind])
        col=str(pd_source.color[ind])
        ms=30
        m.scatter(lon1, lat1,c=col, marker=marker, s=ms, label=val, linestyle = 'None')
        if gdf.Regime[i]=='ablation': # ablation area non-filled symbols
            m.scatter(lon1, lat1,c='#ffff00', marker=marker,  s=ms*0.6)

# !!! want to plot basin outlines and write the name of the basin
fs=20
for basin_index,basin_region in enumerate(basin_regions):
    fn="./ancil/mouginot_sectors/buffer_basins_include_border_regions.shp"
    buffer_basins=gpd.read_file(fn)
#     basins=basins.to_crs('epsg:4326')  #change coord system
#     basins=basins[basins.SUBREGION1==basin_region]
#     #create geodataframe
#     df_scat_geo=gp.GeoDataFrame(df_insitu, geometry= gp.points_from_xy(df_insitu.lon, df_insitu.lat))
#     #search for matching points in basins
#     df_basins = gp.sjoin(basins, gp.GeoDataFrame(df_scat_geo.geometry, crs='epsg:4326'), op='contains')
#     df_basins=df_basins.sort_values(by='index_right')
#     #compare indexes and drop values 
#     df_insitu = df_insitu.loc[df_insitu.index.isin(df_basins.index_right)]
        #label basins
    coords=buffer_basins.geometry.centroid
    names=buffer_basins.SUBREGION1
    for i in range(len(buffer_basins)): 
        xx,yy=m(coords[i].x, coords[i].y)
        print(xx,yy)
        if i==5: yy+=150000; print(yy)
        plt.text(xx, yy, s=names[i], fontsize=fs*1.2, horizontalalignment='center',fontweight='bold', color='white')
        plt.text(xx, yy, s=names[i], fontsize=fs*1.2, horizontalalignment='center', alpha=.8)
show_basins=1
if show_basins==1: 
    #read original dataset
    # fn="./ancil/mouginot_sectors/Greenland_Basins_PS_v1.4.2.shp"
    # basins=gpd.read_file(fn)
    # #buffer shapefile layers to remove geometry issues in original shp
    # buffer_basins = basins.copy()
    # buffer_basins.geometry = buffer_basins['geometry'].buffer(5) #5m is enough
    # buffer_basins=buffer_basins.to_crs('epsg:4326')  #change coord system
    # # merge shapefiles with same subregion
    # basins_diss = buffer_basins.dissolve(by='SUBREGION1')
    # # save buffered and dissolved shp file with lat/lon coord system
    # basins_diss.to_file("./ancil/mouginot_sectors/Greenland_Basins_buff_dissolv_latlon.shp",driver='ESRI Shapefile')
    
    #open created shp in basemap
    fn_new="./ancil/mouginot_sectors/Greenland_Basins_buff_dissolv_latlon"
    m.readshapefile(fn_new, '.shp', linewidth=0.25, color='k', default_encoding='ISO8859-1')

handles, labels = plt.gca().get_legend_handles_labels() #To group labels and only display once
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=11.2)
leg = ax.get_legend()   
leg.set_bbox_to_anchor((.58, 0.245))   #put legend next to map

cbar = plt.colorbar(con,  fraction=.05, pad=0.02) #orientation='horizontal',
cbar.ax.tick_params(labelsize=12)

cbar.set_label("elevation, m",size=15)

if ly =='p':
        fig.savefig('./Figs/program_source_map.png', bbox_inches='tight', dpi=150)
          

#https://stackoverflow.com/questions/26872337/how-can-i-get-my-contour-plot-superimposed-on-a-basemap

# #%%  Scatter of Accumulation vs Elevation
# scatter=1
# if scatter:
#     mult=1
#     plt.rcParams['axes.facecolor'] = 'w'

#     #-------------------------------------------------------data preparation
#     data_scat = pd.DataFrame() 
#     data_scat["Name"] = df_insitu.Name
#     data_scat["lat"] = df_insitu.lat
#     data_scat["lon"] = df_insitu.lon
#     data_scat["Elevation"] = df_insitu.Elevation
#     data_scat["Elevation"][data_scat["Elevation"]==-999]=np.nan
#     data_scat["Accumulation"] = df_insitu.Accumulation
#     data_scat["Ratio"] = data1[:,5] #RCM/obs  --> Careful, values>= 2 are set to 2

#     data_scat["Difference"] = data1[:,6] #RCM/obs  --> Careful, values>= 2 are set to 2
      
#     def getnearpos2(lat,lon, lat_val, lon_val):  #find index of nearest value in array to 'value'
#         X = np.abs(lat-lat_val)
#         Y = np.abs(lon-(lon_val+360))
#         diff = X+Y
#         ij = np.argwhere(diff == diff.min())
#         return ij
    
#     new_elev = np.zeros((data_scat.shape[0],1))*np.nan
#     for i in range(len(data_scat)):
#         if ~np.isnan(df_insitu.lat[i])& ~np.isnan(df_insitu.lon[i]):
#             lat_val = df_insitu.lat[i]
#             lon_val = df_insitu.lon[i]
#             idx = getnearpos2(lat, lon,lat_val, lon_val) 
#             new_elev[i] = elev[idx[0,0],idx[0,1]]
    
#     data_scat["New_elev"] = new_elev
#     data_scat["Diff"] = data_scat.Elevation - data_scat.New_elev
#     data_scat["final_elevation"] = data_scat.Elevation
#     data_scat["Elev2"] = data_scat.New_elev
#     for i in range(len(data_scat)):
#         if np.isnan(data_scat.Elevation[i]):
#             data_scat.final_elevation[i] = data_scat.New_elev[i]
#         if np.isnan(data_scat.Elevation[i]):
#             data_scat.Elev2[i] = np.nan   
#     data_scat["Diff2"] = data_scat.Diff[np.isnan(data_scat.Elevation)]
#     #output df with elev
#     df4=data_scat.drop_duplicates(subset='Name', keep="last")
#     # df4['final_elevation']=df4['final_elevation'].astype(int)
#     df4.rename(columns={'final_elevation': 'elev'}, inplace=True)
#     df4.reset_index(drop=True, inplace=True)

#     df4.to_csv('./ancil/Greenland_sites_for_eccum_rain_evaluation.csv', mode='w', columns=['Name','lat','lon','elev'], float_format='%.4f')
#     #%%
#     # print(np.max(np.abs(data_scat.Diff)))
#     # print(np.mean(np.abs(data_scat.Diff)))
    
#     #---------------------------------------------------------------------plo
#     # fig5 = plt.figure()
#     # plt.scatter(data_scat.Accumulation, data_scat.final_elevation)
#     # #plt.scatter(data_scat.Accumulation, data_scat.Elevation)
#     # n1 = len(data_scat) - np.count_nonzero(np.isnan(data_scat.final_elevation))
#     # plt.text(3100,3100,'N = '+str(n1), fontsize=fs*mult)
#     # plt.ylabel("elevation, m", fontsize=fs*mult)
#     # plt.xlabel("accumulation, mm w.e.", fontsize=fs*mult)
#     # cb_ax = fig5.axes[0] 
#     # cb_ax.tick_params(labelsize=fs*mult) #to adjust tick fontsize
    
#     fig6 = plt.figure()
#     # plt.scatter(data_scat.Ratio, data_scat.final_elevation)
#     plt.scatter(data_scat.Difference, data_scat.final_elevation)
#     n2 = len(data_scat) - np.count_nonzero(np.isnan(data_scat.Ratio))
#     plt.text(1.51,3100,'N = '+str(n2), fontsize=fs*mult)
#     plt.ylabel("elevation, m", fontsize=fs*mult)
#     # plt.xlabel("ratio RCM/obs", fontsize=fs*mult)
#     plt.xlabel('ratio '+RCM_name+'/obs', fontsize=fs*mult)
#     cb_ax2 = fig6.axes[0] 
#     cb_ax2.tick_params(labelsize=fs*mult) #to adjust tick fontsize
#     #labels= ["New elevations","Original elevs where available"]
#     #plt.legend(labels, fontsize=10, loc="upper right")
#     for j in range(0,len(data_scat)):
#         if ((data_scat.final_elevation[j]>2800)&(data_scat.Difference[j]<-250)):
#             plt.text(data_scat.Difference[j], data_scat.final_elevation[j], df_insitu.Name[j]+''+str(df_insitu['End Year'][j]),
#                    fontsize=fs*mult,va='center',rotation=45,ha='left', rotation_mode="anchor")
#         if ((data_scat.final_elevation[j]>0)&(data_scat.Difference[j]<-700)):
#             plt.text(data_scat.Difference[j], data_scat.final_elevation[j], df_insitu.Name[j]+''+str(df_insitu['End Year'][j]),
#                    fontsize=fs*mult,va='center',rotation=45,ha='left', rotation_mode="anchor")
#         if ((data_scat.final_elevation[j]>0)&(data_scat.Difference[j]>250)):
#             plt.text(data_scat.Difference[j], data_scat.final_elevation[j], df_insitu.Name[j]+''+str(df_insitu['End Year'][j]),
#                    fontsize=fs*mult,va='center',rotation=45,ha='left', rotation_mode="anchor")
#         # if (np.abs(data_scat.Diff[j])>200):
#         #     plt.text(data_scat.Accumulation[j],data_scat.Elevation[j], df_insitu.Name[j],
#         #             fontsize=fs*mult,verticalalignment='center',rotation=45)
 
# #%% make gif
# make_gif=1
# if make_gif:
#     figpath='./Figs/tp_vs_insitu/'
#     animpath='/Users/jason/Dropbox/CARRA/CARRA_rain/Figs/tp_vs_insitu/anim/'
#     RCM_names=['CARRA','RACMO2.3p2_5.5km','MARv3.11.5_6km','MARv3.11.5_10km','MARv3.11.5_15km','MARv3.11.5_20km']
#     # RCM_names=['CARRA','RACMO2.3p2_5.5km','MARv3.11.5_6km','MARv3.11.5_10km','MARv3.11.5_15km','MARv3.11.5_20km']
#     RCM_names=['RACMO2.3p2_5.5km']
#     comp_vars=['P','P-E']
#     comp_vars=['P','P-E','P-E-BSS']
#     for comp_var in comp_vars:
#         hivals=[1600,700]
#         hivals=[1600]
#         # for hival in hivals[0:1]:
#         for hival in hivals:
#             print(hival)
#             msg='convert -delay 130 -loop 0 '+\
#                 figpath+RCM_names[0]+'_pits\ and\ cores_tp_vs_insitu_t2m_startdate_'+comp_vars[0]+'_'+str(hival)+'hival.png '+\
#                 figpath+RCM_names[0]+'_pits\ and\ cores_tp_vs_insitu_t2m_startdate_'+comp_vars[1]+'_'+str(hival)+'hival.png '+\
#                 figpath+RCM_names[0]+'_pits\ and\ cores_tp_vs_insitu_t2m_startdate_'+comp_vars[2]+'_'+str(hival)+\
#                     animpath+RCM_names[0]+'_vs_field_data_'+comp_var+'_'+str(hival)+'hival.gif'
#             # msg='convert -delay 130 -loop 0 '+\
#             #     figpath+RCM_names[0]+'_pits\ and\ cores_tp_vs_insitu_t2m_startdate_'+comp_var+'_'+str(hival)+'hival.png '+\
#             #     figpath+RCM_names[1]+'_pits\ and\ cores_tp_vs_insitu_t2m_startdate_'+comp_var+'_'+str(hival)+'hival.png '+\
#             #     figpath+RCM_names[2]+'_pits\ and\ cores_tp_vs_insitu_t2m_startdate_'+comp_var+'_'+str(hival)+'hival.png '+\
#             #     figpath+RCM_names[3]+'_pits\ and\ cores_tp_vs_insitu_t2m_startdate_'+comp_var+'_'+str(hival)+'hival.png '+\
#             #     figpath+RCM_names[4]+'_pits\ and\ cores_tp_vs_insitu_t2m_startdate_'+comp_var+'_'+str(hival)+'hival.png '+\
#             #     figpath+RCM_names[5]+'_pits\ and\ cores_tp_vs_insitu_t2m_startdate_'+comp_var+'_'+str(hival)+'hival.png '+\
#             #         animpath+'RCMs_vs_field_data_'+comp_var+'_'+str(hival)+'hival.gif'
#             print(msg)
#             os.system(msg)
# print("done")