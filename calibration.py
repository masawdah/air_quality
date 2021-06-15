# Built-in modules
import os
import glob
import re
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Basics of Python data handling and visualization
from matplotlib.colors import ListedColormap
import geopandas as gpd
from geopandas import GeoDataFrame as gdf
import rasterio as rio
from shapely import geometry
from rasterstats import zonal_stats
import seaborn as sns
import copy
import numpy as np
from shapely.geometry import Polygon
from rasterio.warp import calculate_default_transform, reproject, Resampling
from typing import List, Union, Tuple, Optional
import pandas as pd
from eolearn.core import EOPatch, EOTask,FeatureType, AddFeature, MapFeatureTask
from eolearn.io import ExportToTiff
from eolearn.features import LinearInterpolation
from collections import defaultdict

import geopandas as gpd
from shapely import wkt
import rasterio as rio
from shapely import geometry
from sentinelhub import bbox_to_dimensions
from pathlib import Path

from utils import (get_extent, 
                   draw_outline, 
                   draw_bbox, 
                   draw_feature, 
                   draw_true_color,
                   unzip_file,
                   load_tiffs,
                   load_list_tiffs,
                   days_to_datetimes,
                   datetimes_to_days,
                   reproject_tiff,
                   upscale_tiff,
                   mask_tiff)

from utils_meoteq import (create_grid,stats_cams,resample_cams_to_target, resample_s5p_to_target,reclassify_land_cover,masking_tiff,
                          reclassify_legend,land_cover_stats,stats_era5,resample_era5_to_target,resample_modis_to_target,extract_data,target_img,
                          stats, filter_MODIS, modis_stats, fill_modis_gaps, export_s5p_band,
                          polygrid, export_geotiff,
                          percs_landuse, upscale_mean_tiff, 
                          extract_points_from_raster, extract_points_from_modis,
                          extract_points_from_timeless_raster,
                          extract_points_from_era5,
                          upscale_nearest_tiff)

# Calibration by forcing averages to be the same
def calibrate(predictions,              # predictions (Geodataframe)
              eopatch,                  # NO2/PM2.5 eopatch to find the boundaries of PM2.5/NO2 product , used to rasterize the final predictions
              dates,                    # target dates
              original_grid,            # Original grid with target pixels centroids
              target_grid,              # Target grid, it will used to rasterize the final predictions
              resampled_dir,            # Path of billinear resampled input NO2/PM2.5 to impute the missing pixels from it before correction and calibration
              original_dir,             # Path of coarse resolution of NO2/PM2.5 (to do calibration & residual correction)
              mask,                     # Use mask to crop the product to the AOI bounding box
              AOI,                      # Area of interest
              model,                    # Model NO2/PM2.5
              sub_dir,                  # Path of submission folder
              viz_dir,                  # Path of visualization folder
              daily_mean_dir=None,      # Path of daily mean cams (to do residual correction)
    
              
             ):
    assert model in ['NO2','PM2_5']
    assert AOI in ['Italy','South_Africa','California']
    if model == 'NO2':
        target_resolution = 1000
        t=1
        perc = 10
        for dd, tt in zip(list(dates.date), list(dates.time)): # Iterate by day and time

            # Prepare df by day - merge predictions
            day_gdf = copy.deepcopy(original_grid)
            day_gdf['date'] = dd
            day_gdf = pd.merge(day_gdf, predictions[['date', 'lon', 'lat', 'pred']], on=['date','lon', 'lat'], how='left')

            # We need to subset the points for each tile ecause the original tiles have small differences in bounding box 
            if AOI == 'South_Africa' or AOI == 'California':
                file_name = 'S5P_NO2__OFFL_L2_day'
            elif AOI == 'Italy':
                file_name = 'S5P_NO2_OFFL_L2_day'
                
            input_filename = f'{file_name}{dd}_T{tt[1:3]}.tif'
            target_img = load_tiffs(original_dir/'NO2', 
                                  (FeatureType.DATA, 'NO2'), 
                                  filename=input_filename) 

            bounds = list(target_img.bbox)
            bounds = geometry.box(bounds[0], bounds[1], bounds[2], bounds[3])
            bounds = gpd.GeoDataFrame(geometry=[bounds], crs="EPSG:4326")
            day_gdf = gpd.overlay(day_gdf, bounds, how='intersection')

            
            # Impute the missing pixels
            res_orig_df = extract_points_from_raster(day_gdf,
                                                     [str(resampled_dir) + f'/NO2/{file_name}'+ str(dd) + '_' + 'T00.tif'], 'res_orig')
            res_orig_df = pd.concat(res_orig_df)
            day_gdf['res_orig'] = list(res_orig_df['res_orig']) 
            day_gdf['pred']= np.where(np.isnan(day_gdf['pred']), day_gdf['res_orig'], day_gdf['pred'])



            # Extract NO2 value of the image to downsample
            orig_df = extract_points_from_raster(day_gdf, 
                                                 [str(original_dir) + f'/NO2/{file_name}'+ str(dd) + '_' + 'T'+ tt[1:3] + '.tif'], 'orig')

            NO_DATA_VALUE= -9999.0
            orig_df = pd.concat(orig_df)
            day_gdf['orig'] = list(orig_df['orig'])
            day_gdf['orig']= np.where(day_gdf['orig'] == NO_DATA_VALUE, day_gdf['pred'], day_gdf['orig'])
            day_gdf['orig']= np.where(day_gdf['orig'] > 1, day_gdf['pred'], day_gdf['orig'])

            # Residual correction
            ## count downscaled pixels related to the origin pixel
            pixl_sums = day_gdf.groupby(by=["ID"]).sum()
            pixl_sums= pixl_sums.rename(columns={"id":"count"})
            pixl_sums= pixl_sums.drop(columns=['lon', 'lat','pred','orig','res_orig']) 
            
            day_gdf = pd.merge(day_gdf, pixl_sums, on=['ID'], how='inner')

            ## avg of downscaled pixels
            pixl_means = day_gdf.groupby(by=["ID"]).mean()
            pixl_means= pixl_means.rename(columns={"pred": "pred_avg", "orig": "orig_avg"})
            pixl_means= pixl_means.drop(columns=['lon', 'lat','count','res_orig'])

            ## Merge & correct
            day_gdf = pd.merge(day_gdf, pixl_means, on=['ID'], how='inner')
            day_gdf['corr_pred'] = day_gdf['pred'] + ((day_gdf['orig'] - day_gdf['pred_avg']) / day_gdf['count'])
            day_gdf['corr_pred'] = np.where(day_gdf['pred'] == day_gdf['res_orig'], day_gdf['pred'], day_gdf['corr_pred'])

            # Calibration
            ## avg of corrected predictions
            pixl_means = day_gdf.groupby(by=["ID"]).mean()
            pixl_means= pixl_means.rename(columns={"corr_pred": "corr_pred_avg"})
            pixl_means= pixl_means.drop(columns=['lon', 'lat','count','orig','pred','pred_avg','res_orig'])

            ## Merge & calibrate
            day_gdf = pd.merge(day_gdf, pixl_means, on=['ID'], how='inner')

            # Calibration by forcing averages to be the same
            day_gdf['cal'] = day_gdf['orig']*(day_gdf['corr_pred']/day_gdf['corr_pred_avg'])

            ## Delete the extreme valueS (low quality pixels), to do the submission without failure !!!!!!!!!
            day_gdf['cal']= np.where(day_gdf['cal'] > 1, np.nan, day_gdf['cal'])
            
            ## fill the low quality (value > 1) with nearest valid pixel
            cal_df = pd.DataFrame(day_gdf['cal'], copy=True)
            cal_df.fillna(method='ffill', inplace=True)
            day_gdf['cal'] = list(cal_df['cal'])

            # rasterize   
            res_down = bbox_to_dimensions(target_img.bbox, target_resolution)
            bounds_down = list(target_img.bbox)
            minx, miny, maxx, maxy = target_img.bbox
            
            
            xres_down = (bounds_down[2]-bounds_down[0])/res_down[0]
            yres_down = (bounds_down[3]-bounds_down[1])/res_down[1]

            sizey = round((maxy-miny)/yres_down)
            sizex = round((maxx-minx)/xres_down)

            transform = rio.transform.from_bounds(minx, miny, maxx, maxy, sizex, sizey)
            shapes = ((geom, value) for geom, value in zip(day_gdf.geometry, day_gdf.cal))
            final_raster = rio.features.rasterize(shapes, out_shape=(sizey, sizex), transform=transform)


            # export for submission
            export_geotiff(str(sub_dir) + f'/{AOI}/NO2/'+ str(dd) + f'_NO2_{AOI}.tif', final_raster, transform, 4326)
            masking_tiff(f'AOIs_bboxes/{AOI}/{mask}-bbox-wgs84.shp', 
                      str(sub_dir) + f'/{AOI}/NO2/'+ str(dd) + f'_NO2_{AOI}.tif', 
                      str(sub_dir) + f'/{AOI}/NO2/'+ str(dd) + f'_NO2_{AOI}.tif')

            # export for visualization
            export_geotiff(str(viz_dir) + f'/{AOI}/NO2/S5P_NO2__OFFL_L2_day' + str(dd) + '_' + 'T' + tt[1:3] + '.tif',
                           final_raster, transform, 4326)
            masking_tiff(f'AOIs_bboxes/{AOI}/{mask}-bbox-wgs84.shp', 
                      str(viz_dir) + f'/{AOI}/NO2/S5P_NO2__OFFL_L2_day' + str(dd) + '_' + 'T' + tt[1:3] + '.tif', 
                      str(viz_dir) + f'/{AOI}/NO2/S5P_NO2__OFFL_L2_day' + str(dd) + '_' + 'T' + tt[1:3] + '.tif') 
            
            print(f"{perc}% Done", end="\r")
            perc+=10
        

    elif model == 'PM2_5':
        if AOI == 'Italy':
            target_resolution = 1000
            t = 1
        elif AOI == 'South_Africa' or AOI == 'California':
            target_resolution = 10000
            t = 10
        perc = 10
           
        for dd, tt in zip(list(dates.date), list(dates.time)): # Iterate by day and time

            # Prepare df by day - merge predictions
            day_gdf = copy.deepcopy(original_grid)
            day_gdf['date'] = dd
            day_gdf = pd.merge(day_gdf, predictions[['date', 'lon', 'lat', 'pred']], on=['date','lon', 'lat'], how='left') 
            
            ## Impute the missing pixels
            res_orig_df = extract_points_from_raster(day_gdf, 
                                                 [str(resampled_dir) + '/PM2_5/CAMS_PM2_5_day'+ str(dd) + '_h00' + '.tif'], 'res_orig')
            res_orig_df = pd.concat(res_orig_df)
            day_gdf['res_orig'] = list(res_orig_df['res_orig']) 
            day_gdf['pred']= np.where(np.isnan(day_gdf['pred']), day_gdf['res_orig'], day_gdf['pred'])         
            
            
            # Extract CAMS value from mean daily scene to do residual correction 
            orig_df = extract_points_from_raster(day_gdf, 
                                                 [str(daily_mean_dir) + '/PM2_5/CAMS_PM2_5_day'+ str(dd) + '_h00' + '.tif'], 'orig')

            orig_df = pd.concat(orig_df)
            day_gdf['orig'] = list(orig_df['orig'])

            # Residual correction
            ## count downscaled pixels related to the origin pixel
            pixl_sums = day_gdf.groupby(by=["ID"]).sum()
            pixl_sums= pixl_sums.rename(columns={"id":"count"})
            pixl_sums= pixl_sums.drop(columns=['lon','date', 'lat','pred','orig','res_orig','index_right'])    
            day_gdf = pd.merge(day_gdf, pixl_sums, on=['ID'], how='inner')

            ## avg of downscaled pixels
            pixl_means = day_gdf.groupby(by=["ID"]).mean()
            pixl_means= pixl_means.rename(columns={"pred": "pred_avg", "orig": "orig_avg"})

            pixl_means= pixl_means.drop(columns=['lon', 'date','lat','count','res_orig','index_right'])

            ## Merge and do the correction
            day_gdf = pd.merge(day_gdf, pixl_means, on=['ID'], how='inner')
            day_gdf['corr_pred'] = day_gdf['pred'] + ((day_gdf['orig'] - day_gdf['pred_avg']) / day_gdf['count'])


            # Calibration - by forcing the average of predicted downscaled pixels to be the same of related origin pixel
            ## Extract CAMS value of the image to downsample
            orig_df_h = extract_points_from_raster(day_gdf, 
                                                 [str(original_dir) + '/PM2_5/CAMS_PM2_5_day'+ str(dd) + '_' + tt + '.tif'], 'orig')
            orig_df_h = pd.concat(orig_df_h)
            day_gdf['orig_h'] = list(orig_df_h['orig'])
            

            ## Compute mean value per pixel
            pixl_m = day_gdf.groupby(by=["ID"]).mean()
            pixl_m = pixl_m.rename(columns={"corr_pred": "corr_pred_avg"})

            pixl_m = pixl_m.drop(columns=['lon', 'date','lat','orig_h','count','res_orig','index_right','pred','orig','pred_avg'])

            ## Merge and calibrate
            day_gdf = pd.merge(day_gdf, pixl_m, on=['ID'], how='inner')
            day_gdf['cal'] = day_gdf['orig_h']*(day_gdf['corr_pred']/day_gdf['pred_avg'])


            # rasterize
            res_down = bbox_to_dimensions(eopatch.bbox, target_resolution)
            bounds_down = list(eopatch.bbox)
            minx, miny, maxx, maxy = target_grid.geometry.total_bounds
            
            xres_down = (bounds_down[2]-bounds_down[0])/res_down[0]
            yres_down = (bounds_down[3]-bounds_down[1])/res_down[1]
            
            sizey = round((maxy-miny)/yres_down)
            sizex = round((maxx-minx)/xres_down)

            transform = rio.transform.from_bounds(minx, miny, maxx, maxy, sizex, sizey)
            shapes = ((geom, value) for geom, value in zip(day_gdf.geometry, day_gdf.cal))
            final_raster = rio.features.rasterize(shapes, out_shape=(sizey, sizex), transform=transform)


            # export for submission
            export_geotiff(str(sub_dir) + f'/{AOI}/PM2.5/'+ str(dd) + f'_PM25_{AOI}.tif', final_raster, transform, 4326)
            masking_tiff(f'AOIs_bboxes/{AOI}/{mask}-bbox-wgs84.shp', 
                      str(sub_dir) + f'/{AOI}/PM2.5/'+ str(dd) + f'_PM25_{AOI}.tif', 
                      str(sub_dir) + f'/{AOI}/PM2.5/'+ str(dd) + f'_PM25_{AOI}.tif')

            # export for visualization
            export_geotiff(str(viz_dir) + f'/{AOI}/PM2.5/CAMS_PM2_5_day' + str(dd) + '_' + tt + '.tif',final_raster, transform, 4326)
            masking_tiff(f'AOIs_bboxes/{AOI}/{mask}-bbox-wgs84.shp', 
                      str(viz_dir) + f'/{AOI}/PM2.5/CAMS_PM2_5_day' + str(dd) + '_' + tt + '.tif', 
                      str(viz_dir) + f'/{AOI}/PM2.5/CAMS_PM2_5_day' + str(dd) + '_' + tt + '.tif')
            
            print(f"{perc}% Done", end="\r")
            perc+=10