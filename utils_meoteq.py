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
import fiona
import rasterio
import rasterio.mask

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

def polygrid(poly, xres, yres, crs):
    """
    Create grid of eopatch (bounds) 
    
    :param poly: bounds of eopatch 
    :param xres: desired pixel's width resolution
    :param yres: desired pixel's hight resolution 
    :param crs: reference corrdinate system 
    
    :return: grid 
    """
    # Get corners
    xmin,ymin,xmax,ymax = poly.total_bounds
    # Define poly vertical and horizontal limits
    cols = list(np.arange(xmin, xmax, xres))
    rows = list(np.arange(ymin, ymax, yres))
    rows.reverse()
    # Create polygons
    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(Polygon([(x,y), (x+xres, y), (x+xres, y+yres), (x, y+yres)]))
    # Return gpd
    grid = gpd.GeoDataFrame({'geometry':polygons}, crs=crs)
    return grid


def create_grid(eopatch:EOPatch, model:str, AOI:str):
    """
    Create grid and centroids of eopatch (bounds) at orignal and target resolution
    
    :param eopatch: bounds of eopatch 
    :param model: desired pixel's width resolution
    :param AOI: desired pixel's hight resolution 
    
    :return: original grid, original centroids, target grid, target centroids 
    """
    assert AOI in ['Italy','California','South_Africa']
    assert model in ['NO2','PM2_5']
    
    if model == 'NO2':
        target_resolution = 1000
    else:
        if AOI == 'Italy':
            target_resolution = 1000
        else:
            target_resolution = 10000

    # Create original rectangular grid for pixels corresponding to NO2/PM2.5
    bounds = list(eopatch.bbox)
    
    xres = (bounds[2]-bounds[0])/eopatch.data[model].shape[1]
    yres = (bounds[3]-bounds[1])/eopatch.data[model].shape[2]
    
    bounds = geometry.box(bounds[0], bounds[1], bounds[2], bounds[3])
    bounds = gpd.GeoDataFrame({"id":1,"geometry":[bounds]}, crs="EPSG:4326")
    grid = polygrid(bounds, xres,yres, 4326)
    centroids = gpd.GeoDataFrame(geometry= grid.centroid)
    
    # Create downscaled rectangular grid (target) for pixels corresponding to NO2/PM2.5
    bounds_down = list(eopatch.bbox)
    res_down = bbox_to_dimensions(eopatch.bbox, target_resolution)
    
    xres_down = (bounds_down[2]-bounds_down[0])/res_down[0]
    yres_down = (bounds_down[3]-bounds_down[1])/res_down[1]
    
    bounds_down = geometry.box(bounds_down[0], bounds_down[1], bounds_down[2], bounds_down[3])
    bounds_down = gpd.GeoDataFrame({"id":1,"geometry":[bounds_down]}, crs="EPSG:4326")
    grid_down = polygrid(bounds_down, xres_down,yres_down, 4326)
    centroids_down = gpd.GeoDataFrame(geometry= grid_down.centroid)

    return grid, centroids, grid_down, centroids_down

def reclassify_legend(AOI:str):
    """
    Visulize the landcover
    
    :param AOI: area of intrest 
    :return: color ramp, legend explians the color 
    """
        
    assert AOI in ['Italy','California','South_Africa']
    if AOI == 'California':
        # Define the colors
        cmap = ListedColormap(["white", "red", "yellow", "green","blue"])
        # Add a legend for labels
        legend_labels = {"white": "NODATA", 
                         "red": "urban",
                         "yellow": "agriculture",
                         "green": "natural",
                         "blue": "water",
                         }
        
    elif AOI == 'South_Africa':
        # Define the colors
        cmap = ListedColormap(["white", "red", "purple", "yellow", "green"])
        # Add a legend for labels
        legend_labels = {"white": "NODATA", 
                         "red": "urban",
                         "purple": "indtrans",
                         "yellow": "agriculture",
                         "green": "natural"             
                         }

    elif AOI == 'Italy':
        # Define the colors
        cmap = ListedColormap(["white", "salmon", "brown", "yellow", "green", "lightblue"])
        # Add a legend for labels
        legend_labels = {"white": "NODATA", 
                         "salmon": "urban",
                         "brown": "industry and transport",
                         "yellow": "agriculture",
                         "green": "natural",
                         "lightblue": "water"}
        

        
    return cmap, legend_labels

def reclassify_land_cover(eopatch:EOPatch, feature_name:str, AOI:str,land_cover_path:str,land_cover_name:str, out_name:str,number_of_classes:int):
    """
    Reclassify the land cover
    
    :param eopatch: EOPatch for the land cover
    :param feature_name: Name of mask timeless data (land cover layer) in the eopatch.
    :param AOI: Area of intrest
    :param land_cover_path: Directory file where land cover layer is stored 
    :param land_cover_name: Name of land cover raster
    :param out_name: Name of output reclassified raster
    :param number_of_classes: Number of reclassifed classes
    
    :return: None
    """
    
    assert AOI in ['Italy','California','South_Africa']
    land_cover_path = str(land_cover_path)
    
    if AOI == 'California':
        # Reclassify values
        array = eopatch.mask_timeless[feature_name]
        eopatch.mask_timeless[feature_name][np.isin(array, [20,30,60,70,90,111,116,121])] = 3 # Natural
        eopatch.mask_timeless[feature_name][np.isin(array, [40])] = 2 # Agriculture
        eopatch.mask_timeless[feature_name][np.isin(array, [50])] = 1 # Urban
        eopatch.mask_timeless[feature_name][np.isin(array, [80,200])] = 4 # water

    elif AOI == 'South_Africa':
        # Reclassify values
        array = eopatch.mask_timeless[feature_name]
        eopatch.mask_timeless[feature_name][np.isin(array, list(range(1,32)))] = 4 # Natural
        eopatch.mask_timeless[feature_name][np.isin(array, list(range(32,47)))] = 3 # Agriculture
        eopatch.mask_timeless[feature_name][np.isin(array, list(range(47,66)))] = 1 # Urban
        eopatch.mask_timeless[feature_name][np.isin(array, list(range(66,74)))] = 2 # Industrial, roads, rail, mines, landfills

    elif AOI == 'Italy':
        array = eopatch.mask_timeless[feature_name]
        eopatch.mask_timeless[feature_name][array==128] = 0 # no value
        eopatch.mask_timeless[feature_name][np.isin(array, [1,2,11])] = 1 # Urban fabric
        eopatch.mask_timeless[feature_name][np.isin(array, list(range(3,10)))] = 2 # Industry and transport
        eopatch.mask_timeless[feature_name][np.isin(array, list(range(12,23)))] = 3 # Agriculture
        eopatch.mask_timeless[feature_name][array == 10] = 4 # Natural
        eopatch.mask_timeless[feature_name][np.isin(array, list(range(23,40)))] = 4 # Natural
        eopatch.mask_timeless[feature_name][np.isin(array, list(range(40,45)))] = 5 # Water
        #eopatch.mask_timeless[feature_name][np.isin(array, [0])] = 5 # Water

    src = rio.open(land_cover_path + '/' + land_cover_name)
    affine = src.transform
    export_geotiff(land_cover_path + '/'+ out_name +'.tif', eopatch.mask_timeless[feature_name][:, :, 0], affine, 4326)

    
def _percs_landuse_usa(zstats, minpixels):
    """Function to do zonal statistcs for California land cover - percentage of each land cover inside each pixel """

    empty = []
    urban = []
    agri = []
    natural = []
    water = []
    
    for indx, vals in enumerate(zstats):
    
        # Does it reach min?
        if vals['count']<minpixels:
            empty.append(np.nan)
            urban.append(np.nan)
            agri.append(np.nan)
            natural.append(np.nan)
            water.append(np.nan)
        else:
            # Empty
            if 0 in vals:
                empty.append(vals[0]/vals['count'])
            else:
                empty.append(0)
            # Urban
            if 1 in vals:
                urban.append(vals[1]/vals['count'])
            else:
                urban.append(0)
            # Agriculture
            if 2 in vals:
                agri.append(vals[2]/vals['count'])
            else:
                agri.append(0)    
            # Natural areas
            if 3 in vals:
                natural.append(vals[3]/vals['count'])
            else:
                natural.append(0)    
            # water
            if 4 in vals:
                water.append(vals[4]/vals['count'])
            else:
                water.append(0)
                
    # Set to nan if mostly empty
    urban = [urban[i] if empty[i]<0.5 else np.nan for i in range(len(urban))]
    agri = [agri[i] if empty[i]<0.5 else np.nan for i in range(len(agri))]
    natural = [natural[i] if empty[i]<0.5 else np.nan for i in range(len(natural))]
    water = [water[i] if empty[i]<0.5 else np.nan for i in range(len(water))]

    return urban, agri, natural, water
    
def _percs_landuse_sa(zstats, minpixels):
    """Function to do zonal statistcs for South Africa land cover - percentage of each land cover inside each pixel """

    empty = []
    urban = []
    indtrans = []
    agri = []
    natural = []
    
    for indx, vals in enumerate(zstats):
    
        # Does it reach min?
        if vals['count']<minpixels:
            empty.append(np.nan)
            urban.append(np.nan)
            indtrans.append(np.nan)
            agri.append(np.nan)
            natural.append(np.nan)
        else:
            # Empty
            if 0 in vals:
                empty.append(vals[0]/vals['count'])
            else:
                empty.append(0)
            # Urban
            if 1 in vals:
                urban.append(vals[1]/vals['count'])
            else:
                urban.append(0)
            # Industry and transport
            if 2 in vals:
                indtrans.append(vals[2]/vals['count'])
            else:
                indtrans.append(0)    
            # Agriculture
            if 3 in vals:
                agri.append(vals[3]/vals['count'])
            else:
                agri.append(0)    
            # Natural areas
            if 4 in vals:
                natural.append(vals[4]/vals['count'])
            else:
                natural.append(0)
                
    # Set to nan if mostly empty
    urban = [urban[i] if empty[i]<0.5 else np.nan for i in range(len(urban))]
    indtrans = [indtrans[i] if empty[i]<0.5 else np.nan for i in range(len(indtrans))]
    agri = [agri[i] if empty[i]<0.5 else np.nan for i in range(len(agri))]
    natural = [natural[i] if empty[i]<0.5 else np.nan for i in range(len(natural))]

    return urban, indtrans, agri, natural

def _percs_landuse_italy(zstats, minpixels):
    """Function to do zonal statistcs for Italy land cover - percentage of each land cover inside each pixel """

    empty = []
    urban = []
    indtrans = []
    agri = []
    natural = []
    water = []
    
    for indx, vals in enumerate(zstats):
    
        # Does it reach min?
        if vals['count']<minpixels:
            empty.append(np.nan)
            urban.append(np.nan)
            indtrans.append(np.nan)
            agri.append(np.nan)
            natural.append(np.nan)
            water.append(np.nan)
        else:
            # Empty
            if 0 in vals:
                empty.append(vals[0]/vals['count'])
            else:
                empty.append(0)
            # Urban
            if 1 in vals:
                urban.append(vals[1]/vals['count'])
            else:
                urban.append(0)
            # Industry and transport
            if 2 in vals:
                indtrans.append(vals[2]/vals['count'])
            else:
                indtrans.append(0)    
            # Agriculture
            if 3 in vals:
                agri.append(vals[3]/vals['count'])
            else:
                agri.append(0)    
            # Natural areas
            if 4 in vals:
                natural.append(vals[4]/vals['count'])
            else:
                natural.append(0)

            if 5 in vals:
                water.append(vals[5]/vals['count'])
            else:
                water.append(0)
                
    # Set to nan if mostly empty
    urban = [urban[i] if empty[i]<0.5 else np.nan for i in range(len(urban))]
    indtrans = [indtrans[i] if empty[i]<0.5 else np.nan for i in range(len(indtrans))]
    agri = [agri[i] if empty[i]<0.5 else np.nan for i in range(len(agri))]
    natural = [natural[i] if empty[i]<0.5 else np.nan for i in range(len(natural))]
    water = [water[i] if empty[i]<0.5 else np.nan for i in range(len(water))]

    return urban, indtrans, agri, natural, water

LC_PERCS = dict(Italy= _percs_landuse_italy,
                California = _percs_landuse_usa,
                    South_Africa = _percs_landuse_sa)

def percs_landuse(zstats, minpixels, AOI):
    assert AOI in ['Italy','California', 'South_Africa']
    if AOI == 'South_Africa':
        urban, indtrans, agri, natural = LC_PERCS[AOI](zstats, minpixels)
        return urban, indtrans, agri, natural
    elif AOI == 'Italy':
        urban, indtrans, agri, natural, water = LC_PERCS[AOI](zstats, minpixels)
        return urban, indtrans, agri, natural, water
    elif AOI == 'California':
        urban, agri, natural, water = LC_PERCS[AOI](zstats, minpixels)
        return urban, agri, natural, water

def land_cover_stats(eopatch:EOPatch,feature_name:str, target_grid, target_centroids, target_bbox, AOI:str, model:str, land_cover_path:str, land_cover_name:str, out_path:str):
    """
    Zonal statisics for perecntage land cover classes in each pixel
    
    :param eopatch: EOPatch for reclassified landcover
    :param feature_name: Name of mask timeless data (reclassified land cover layer) in the eopatch
    :param target_grid: Grid of training data at target resolution
    :param target_bbox: Directory file where land cover layer is stored 
    :param AOI: Name of land cover raster
    :param land_cover_path: Name of output reclassified raster
    :param land_cover_name: Number of reclassifed classes
    :param out_path: Number of reclassifed classes
    
    :return: None
    """ 
    assert AOI in ['Italy','California','South_Africa']
    assert model in ['NO2','PM2_5']

    if model == 'NO2':
        target_resolution = 1000
        t=1
        buffer_value = 50
    else:
        if AOI == 'Italy':
            target_resolution = 1000
            t = 1
            buffer_value = 50
        else:
            target_resolution = 10000
            t = 10
            buffer_value = 5


    land_cover_path = str(land_cover_path)
    src = rio.open(land_cover_path + '/' + land_cover_name)
    affine = src.transform

    array = eopatch.mask_timeless[feature_name]
    array = array[:, :, 0] # Remove one dimension

    zstats = zonal_stats(target_grid, array, affine=affine, stats="count", categorical=True)
    count_pixels = []
    for zst in zstats:
        count_pixels.append(zst["count"])
    min_pixels = 0.75*max(count_pixels)

    res_down = bbox_to_dimensions(target_bbox, target_resolution)
    bounds = list(target_bbox)
    xres = (bounds[2]-bounds[0])/res_down[0]
    yres = (bounds[3]-bounds[1])/res_down[1]
        
    # Rasterize
    LC_grid = copy.deepcopy(target_grid)
    minx, miny, maxx, maxy = LC_grid.geometry.total_bounds
    sizey = round((maxy-miny)/yres)
    sizex = round((maxx-minx)/xres)
    transform = rio.transform.from_bounds(minx, miny, maxx, maxy, sizex, sizey)

    if AOI == 'California':
        urban, agri, natural, water = percs_landuse(zstats, minpixels=min_pixels, AOI=AOI)
        
        LC_grid[f'urban_{t}km'] = urban
        LC_grid[f'agri_{t}km'] = agri
        LC_grid[f'natural_{t}km'] = natural
        LC_grid[f'water_{t}km'] = water
        
        perc = 30
        for lc_class in [f'urban_{t}km', f'agri_{t}km', f'natural_{t}km',f'water_{t}km']:
            shapes = ((geom, value) for geom, value in zip(LC_grid.geometry, LC_grid[lc_class]))
            lc = rio.features.rasterize(shapes, out_shape=(sizey, sizex), transform=transform)
            export_geotiff(out_path + f'/{feature_name}_'+ model + '_' + lc_class +'.tif', lc, transform, 4326)
            print(f"{perc}% Done", end="\r")
            perc+=10

        buffs = copy.deepcopy(target_centroids)
        buffs['geometry'] = buffs.geometry.buffer(xres * buffer_value)

        for lc_class in ['urban','agri', 'natural', 'water']:
            # Open dataset
            src = rio.open(out_path + f'/{feature_name}_'+ model + '_' + lc_class +f'_{t}km.tif')
            affine = src.transform
            land_cover_eop = load_tiffs(datapath=Path(out_path),
                                        feature=(FeatureType.MASK_TIMELESS, feature_name), 
                                        filename=f'{feature_name}_'+ model + '_' + lc_class +f'_{t}km.tif',
                                        image_dtype=np.float32,
                                        no_data_value=9999)
            array = land_cover_eop.mask_timeless[feature_name]
            array = array[:, :, 0] # Remove one dimension
            zstats = zonal_stats(buffs, array, affine=affine, stats="mean", nodata=np.nan)
            vals = []
            for index, value in enumerate(zstats):
                vals.append(value['mean'])

            # Add to grid
            LC_grid[lc_class + '_50km'] = vals
            
            # Rasterize
            shapes = ((geom, value) for geom, value in zip(LC_grid.geometry, LC_grid[lc_class + '_50km']))
            lc = rio.features.rasterize(shapes, out_shape=(sizey, sizex), transform=transform)
            export_geotiff(out_path + f'/{feature_name}_'+ model + '_' + lc_class + '_50km' + '.tif', lc, transform, 4326)
            print(f"{perc}% Done", end="\r")
            perc+=10
            

    elif AOI == 'South_Africa':
        urban, indtrans, agri, natural = percs_landuse(zstats, minpixels=min_pixels, AOI=AOI)
        
        LC_grid = copy.deepcopy(target_grid)
        LC_grid[f'urban_{t}km'] = urban
        LC_grid[f'indtrans_{t}km'] = indtrans
        LC_grid[f'agri_{t}km'] = agri
        LC_grid[f'natural_{t}km'] = natural
        
        perc = 30
        for lc_class in [f'urban_{t}km', f'indtrans_{t}km', f'agri_{t}km',f'natural_{t}km']:
            shapes = ((geom, value) for geom, value in zip(LC_grid.geometry, LC_grid[lc_class]))
            lc = rio.features.rasterize(shapes, out_shape=(sizey, sizex), transform=transform)
            export_geotiff(out_path + f'/{feature_name}_'+ model + '_' + lc_class +'.tif', lc, transform, 4326)
            print(f"{perc}% Done", end="\r")
            perc+=10
            
        buffs = copy.deepcopy(target_centroids)
        buffs['geometry'] = buffs.geometry.buffer(xres * buffer_value)

        for lc_class in ['urban','indtrans', 'agri', 'natural']:
            # Open dataset
            src = rio.open(out_path + f'/{feature_name}_'+ model + '_' + lc_class +f'_{t}km.tif')
            affine = src.transform
            land_cover_eop = load_tiffs(datapath=Path(out_path),
                                        feature=(FeatureType.MASK_TIMELESS, feature_name), 
                                        filename=f'{feature_name}_'+ model + '_' + lc_class +f'_{t}km.tif',
                                        image_dtype=np.float32,
                                        no_data_value=9999)
            array = land_cover_eop.mask_timeless[feature_name]
            array = array[:, :, 0] # Remove one dimension
            zstats = zonal_stats(buffs, array, affine=affine, stats="mean", nodata=np.nan)
            vals = []
            for index, value in enumerate(zstats):
                vals.append(value['mean'])

            # Add to grid
            LC_grid[lc_class + '_50km'] = vals
            
            # Rasterize
            shapes = ((geom, value) for geom, value in zip(LC_grid.geometry, LC_grid[lc_class + '_50km']))
            lc = rio.features.rasterize(shapes, out_shape=(sizey, sizex), transform=transform)
            export_geotiff(out_path + f'/{feature_name}_'+ model + '_' + lc_class + '_50km' + '.tif', lc, transform, 4326)
            print(f"{perc}% Done", end="\r")
            perc+=10


    elif AOI == 'Italy':
        urban, indtrans, agri, natural, water = percs_landuse(zstats, minpixels=min_pixels, AOI=AOI)
        
        LC_grid = copy.deepcopy(target_grid)
        LC_grid[f'urban_{t}km'] = urban
        LC_grid[f'indtrans_{t}km'] = indtrans
        LC_grid[f'agri_{t}km'] = agri
        LC_grid[f'natural_{t}km'] = natural
        LC_grid[f'water_{t}km'] = water
        
        perc = 10
        for lc_class in [f'urban_{t}km', f'indtrans_{t}km', f'agri_{t}km',f'natural_{t}km',f'water_{t}km']:
            shapes = ((geom, value) for geom, value in zip(LC_grid.geometry, LC_grid[lc_class]))
            lc = rio.features.rasterize(shapes, out_shape=(sizey, sizex), transform=transform)
            export_geotiff(out_path + f'/{feature_name}_'+ model + '_' + lc_class +'.tif', lc, transform, 4326)
            print(f"{perc}% Done", end="\r")
            perc+=10
            
        buffs = copy.deepcopy(target_centroids)
        buffs['geometry'] = buffs.geometry.buffer(xres * buffer_value)

        for lc_class in ['urban','indtrans', 'agri', 'natural','water']:
            # Open dataset
            src = rio.open(out_path + f'/{feature_name}_'+ model + '_' + lc_class +f'_{t}km.tif')
            affine = src.transform
            land_cover_eop = load_tiffs(datapath=Path(out_path),
                                        feature=(FeatureType.MASK_TIMELESS, feature_name), 
                                        filename=f'{feature_name}_'+ model + '_' + lc_class +f'_{t}km.tif',
                                        image_dtype=np.float32,
                                        no_data_value=9999)
            array = land_cover_eop.mask_timeless[feature_name]
            array = array[:, :, 0] # Remove one dimension
            zstats = zonal_stats(buffs, array, affine=affine, stats="mean", nodata=np.nan)
            vals = []
            for index, value in enumerate(zstats):
                vals.append(value['mean'])

            # Add to grid
            LC_grid[lc_class + '_50km'] = vals
            
            # Rasterize
            shapes = ((geom, value) for geom, value in zip(LC_grid.geometry, LC_grid[lc_class + '_50km']))
            lc = rio.features.rasterize(shapes, out_shape=(sizey, sizex), transform=transform)
            export_geotiff(out_path + f'/{feature_name}_'+ model + '_' + lc_class + '_50km' + '.tif', lc, transform, 4326)
            print(f"{perc}% Done", end="\r")
            perc+=10

    return LC_grid

def export_geotiff(path, rast, trans, epsg):
    """
    Export array as geotiff raster 
    Taken from https://rasterio.readthedocs.io/en/latest/api
    """    
    new_dataset = rio.open(path, 'w', driver='GTiff',
                           height = rast.shape[0], width = rast.shape[1],
                           count=1, dtype=str(rast.dtype),
                           crs="EPSG:"+str(epsg),
                           transform=trans)
    new_dataset.write(rast, 1)
    new_dataset.close()
    
def upscale_mean_tiff(input_filename: str, output_filename: str, out_shape: Tuple[int, int]):
    """ 
    Upscale a tiff file given a target shape using average resampling. Writes out first channel only
    """
    
    with rio.open(input_filename) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                out_shape[0],
                out_shape[1]
            ),
            resampling=Resampling.average)

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        
        out_meta = dataset.meta
        
    out_meta.update({"driver": "GTiff",
                     "height": data.shape[1],
                     "width": data.shape[2],
                     "transform": transform,
                     "count": 1})

    with rio.open(output_filename, 'w', **out_meta) as dest:
        dest.write(data[:1])
        
def upscale_nearest_tiff(input_filename: str, output_filename: str, out_shape: Tuple[int, int]):
    """ Upscale a tiff file given a target shape using nearest neighbour resampling. Writes out first channel only """
    
    with rio.open(input_filename) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                out_shape[0],
                out_shape[1]
            ),
            resampling=Resampling.nearest)

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        
        out_meta = dataset.meta
        
    out_meta.update({"driver": "GTiff",
                     "height": data.shape[1],
                     "width": data.shape[2],
                     "transform": transform,
                     "count": 1})

    with rio.open(output_filename, 'w', **out_meta) as dest:
        dest.write(data[:1])
        



def ValidData(eopatch,band,min_modis_qa):
    """
    Masked the low quality MODIS pixels 
    
    :param eopatch: Eopatch where MODIS data are stored  
    :param band: Bands number in the eopatch 
    :param min_modis_qa: Threshold quality assurance value
    
    :return: Masked array of MODIS band
    """ 
    eopatch.data['AOD_QA'][..., band][eopatch.data['AOD_QA'][..., band] == min_modis_qa] = np.nan
    modis_AOD_QA = eopatch.data['AOD_QA'][..., band]

    qa_data = np.unique(modis_AOD_QA)
    qa_data = qa_data.astype(np.int16)
    qa_data = qa_data.tolist()
    qa_data = [x for x in qa_data if x != 0]

    for i in qa_data:
        mask811 = 0b111100000000
        mask02 =  0b000000000111
        qa811 = (i & mask811) >> 8
        qa02 = (i & mask02) >> 0
        if qa811 == 0 and qa02 == 1:
            i = float(i)
            modis_AOD_QA=np.where(modis_AOD_QA==i, 0, modis_AOD_QA)
        else:
            i = float(i)
            modis_AOD_QA=np.where(modis_AOD_QA==i, 1, modis_AOD_QA)


    return modis_AOD_QA

def filter_MODIS(modis_eops, modis_qa_eops):
    """
    Filter the low quality MODIS pixels and add new filtered band to the eopatch
    
    :param modis_eops: Eopatch where MODIS bands are stored  
    :param modis_qa_eops: Eopatch where QA bands are stored  
    
    :return: None
    """ 
    feature = (FeatureType.DATA, 'AOD_Valid')
    add_feature = AddFeature(feature)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for index, modis_eop in enumerate(modis_eops):
            ## Call the AOD_QA EOpatch for the same AOD EOpatch
            modis_qa_eop = modis_qa_eops[index]
            
            ## Check how many bands in the AOD eopatch
            modis_AOD = modis_eop.data['AOD']
            t, w, h, b = modis_AOD.shape
            
            ## Loop through bands
            Masked_data=[]
            for band in range(b):
                ## Take the band in the AOD Eopatch
                MODIS_AOD = modis_eop.data['AOD'][..., band]
                
                ## Creat mask of clear and high quality pixels for the band in AOD Eopatch from MODIS QA Eopatch
                valmask = ValidData(modis_qa_eop,band,min_modis_qa = 0)

                ## Creat Masked Array of MODIS AOD
                Masked_MODIS_AOD = np.ma.array(MODIS_AOD, mask=valmask,fill_value=np.nan)
                
                ## Put the maske array in a list 
                Masked_data.append(Masked_MODIS_AOD)
            
            ## Stack the mask arrays for the availabe bands into one array 
            data = np.ma.stack(Masked_data, axis=-1)
            
            ## Add the masked arrays as data in the AOD Eopatch 
            modis_eop = add_feature.execute(modis_eop, data)


def modis_stats(modis_eops,path):
    """
    Compute Statistics (mean, minimum, and maximum) to derive a single daily observation and export a new single daily tiff
    
    :param modis_eops: Eopatch where MODIS bands are stored  
    :param path: Path of directory where to save the new single daily observation  
    
    :return: None
    """ 
    MODIS_NO_DATA_VALUE = -28672

    ## Tasks to compute statistics of valid AOD (mean, maximum, minimum) 
    mean = MapFeatureTask((FeatureType.DATA,'AOD_Valid'),  # input features
                             (FeatureType.DATA_TIMELESS,'Stats_MODIS'),  # output feature
                             np.nanmean,                    # a function to apply to each feature
                             axis=-1)                  


    maximum = MapFeatureTask((FeatureType.DATA,'AOD_Valid'),  # input features
                             (FeatureType.DATA_TIMELESS,'Stats_MODIS'),  # output feature
                             np.nanmax,                    # a function to apply to each feature
                             axis=-1)                   

    minimum = MapFeatureTask((FeatureType.DATA,'AOD_Valid'),  # input features
                             (FeatureType.DATA_TIMELESS,'Stats_MODIS'),  # output feature
                             np.nanmin,                    # a function to apply to each feature
                             axis=-1)
    ## Task to export the merged tiff (mean of daily measurments)
    export_tiff = ExportToTiff((FeatureType.DATA_TIMELESS, 'Stats_MODIS'))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for modis_eop in modis_eops:
            modis_eop.data['AOD_Valid'][modis_eop.data['AOD_Valid'] == MODIS_NO_DATA_VALUE] = np.nan
            tiffname = modis_eop.meta_info['Names'][0]
            t,w,h,b = modis_eop.data['AOD'].shape
            for stats, file_name in [(mean, 'daily_mean_AOD'), (maximum, 'daily_maximum_AOD'), (minimum, 'daily_minimum_AOD')]:
                stats(modis_eop)
                filled_values = modis_eop.data_timeless['Stats_MODIS'][0]
                filled_values.data[filled_values.data == 1.e+20] = np.nan
                modis_eop.data_timeless['Stats_MODIS'][0] = filled_values
                modis_eop.data_timeless['Stats_MODIS'] = np.resize(modis_eop.data_timeless['Stats_MODIS'],(w, h,1))
                export_tiff.execute(modis_eop, filename=str(path)+'/'+file_name+'/'+tiffname)

def fill_modis_gaps(modis_eop, modis_products: list, path):
    """
    Fill the gap in MODIS data
    
    :param modis_eops: Eopatch where MODIS data are stored  
    :param path: Path of directory where save the new filled MODIS data  
    
    :return: None
    """     
    feature = (FeatureType.DATA_TIMELESS, 'filled_AOD')
    add_feature = AddFeature(feature)

    for modis_p in modis_products:
        linear_interp = LinearInterpolation(modis_p)
        linear_interp(modis_eop)
    
    ## Loop into the patch and sign the mean value of MODIS observations to the emaining missing pixels
    mean_AOD = modis_eop.data['daily_mean_AOD']
    t, w, h, _ = mean_AOD.shape

    for i in range(t):
        for modis_p in modis_products:
            filled_values = modis_eop.data[modis_p][i]
            filled_values[np.isnan(filled_values)] = np.nanmean(filled_values)
            modis_eop.data[modis_p][i] = filled_values

    for i in range(t):
        tiffname = modis_eop.meta_info['tiff_names'][i]
        for modis_p in modis_products:
            ## extract the band
            data = modis_eop.data[modis_p][i]
            add_feature.execute(modis_eop,data)
            
            ## Task to export the band (mean of daily measurments)
            export_tiff = ExportToTiff((FeatureType.DATA_TIMELESS, 'filled_AOD'))
            export_tiff.execute(modis_eop, filename=str(path)+'/'+modis_p+'/'+tiffname)

def export_s5p_band(s5p_eop, s5p_products:list, path):
    """
    Export S5P bands to same bounding box to make sure they are coincide 
    
    :param s5p_eop: Eopatch where S5P data are stored  
    :param s5p_products: S5P products to be extracted (NO2, UV Aerosol Index)  
    :param s5p_products: Path of directory where save the extracted S5P bands  
    
    :return: None
    """   
    ## Task to add the filtered NO2 band 
    add_s5p = (FeatureType.DATA_TIMELESS, 'S5P')
    add_s5p_band = AddFeature(add_s5p)

    ## Task to export the merged tiff (mean of daily measurments)
    export_tiff = ExportToTiff((FeatureType.DATA_TIMELESS, 'S5P'))


    for s5p_p in s5p_products:
        t,x,y,b = s5p_eop.data[s5p_p].shape
        for index in range(t):
            no2_array = s5p_eop.data[s5p_p][index][...,0]
            # add dimension
            no2_array= np.expand_dims(no2_array, axis=-1)

            ## execute the add task
            add_s5p_band.execute(s5p_eop,no2_array)

            ## tiff name (filtered NO2)
            tiffname = s5p_eop.meta_info['Names_' + s5p_p][index]

            ## export the tiff (filtered NO2)
            export_tiff.execute(s5p_eop, filename=str(path)+'/'+ s5p_p + '/' +tiffname)
        

def target_img(target_dates, AOI:str, model:str):
    """
    Select the names of target images and the names of predictors images 
    
    :param target_dates: Eopatch where S5P data are stored  
    :param AOI: Area of interest
    :param model: Model (PM 2.5 , NO2) 
    
    :return: None
    """   
    assert AOI in ['Italy','South_Africa','California']
    assert model in ['NO2','PM2_5']

    ## Extract target tiff name 
    CAMS_PM25_tiffnames = []
    CAMS_NO2_tiffnames = []
    S5P_NO2_tiffnames = []
    uv_aerosol_tiffnames = []
    modis_tiffnames = []
    era5_tiffnames = [[],[],[],[]]

    if model == 'PM2_5':
        era5_products = ['rh', 'srwc', 'u', 'v']
        for index in target_dates.index:
            day, h = target_dates.iloc[index]
            CAMS_tiffname = f'CAMS_PM2_5_day{day}_h00.tif'
            CAMS_NO2_tiffname = f'CAMS_NO2_day{day}_h00.tif'
            uv_aerosol_tiffname = f'S5P_AER_AI_OFFL_L2_day{day}_T00.tif'
            modis_tiffname = f'MCD19A2_day{day}.tif'
            
            if AOI == 'Italy':
                S5P_NO2_tiffname = f'S5P_NO2_OFFL_L2_day{day}_T00.tif'
            else:
                S5P_NO2_tiffname = f'S5P_NO2__OFFL_L2_day{day}_T00.tif'
                
            for index, era5_p in enumerate(era5_products):
                era5_tiffname =  f'ERA5_{era5_p}_day{day-1}_h00.tif'
                era5_tiffnames[index].append(era5_tiffname)
                era5_tiffname =  f'ERA5_{era5_p}_day{day}_h00.tif'
                era5_tiffnames[index].append(era5_tiffname)
                
            CAMS_PM25_tiffnames.append(CAMS_tiffname)
            CAMS_NO2_tiffnames.append(CAMS_NO2_tiffname)
            S5P_NO2_tiffnames.append(S5P_NO2_tiffname)
            uv_aerosol_tiffnames.append(uv_aerosol_tiffname)
            modis_tiffnames.append(modis_tiffname)

    elif model == 'NO2':
        era5_products = ['rh', 'srwc', 'u', 'v']

        for index in target_dates.index:
            day, h = target_dates.iloc[index]
            CAMS_tiffname = f'CAMS_PM2_5_day{day}_h00.tif'
            CAMS_NO2_tiffname = f'CAMS_NO2_day{day}_h00.tif'
            aerosol_tiffnme = f'S5P_AER_AI_OFFL_L2_day{day}_T00.tif'
            modis_name= f'MCD19A2_day{day}.tif'
            
            if AOI == 'Italy':
                S5P_NO2_tiffname = f'S5P_NO2_OFFL_L2_day{day}_T00.tif'
            else:
                S5P_NO2_tiffname = f'S5P_NO2__OFFL_L2_day{day}_T00.tif'
                
            
            for index, era5_p in enumerate(era5_products):
                era5_tiffname =  f'ERA5_{era5_p}_day{day-1}_h00.tif'
                era5_tiffnames[index].append(era5_tiffname)
                era5_tiffname =  f'ERA5_{era5_p}_day{day}_h00.tif'
                era5_tiffnames[index].append(era5_tiffname)
                
            S5P_NO2_tiffnames.append(S5P_NO2_tiffname)
            uv_aerosol_tiffnames.append(aerosol_tiffnme)
            CAMS_NO2_tiffnames.append(CAMS_NO2_tiffname)
            CAMS_PM25_tiffnames.append(CAMS_tiffname)
            modis_tiffnames.append(modis_name)

    all_tiffnames = []
    all_tiffnames.append(CAMS_PM25_tiffnames)
    all_tiffnames.append(CAMS_NO2_tiffnames)
    all_tiffnames.append(S5P_NO2_tiffnames)
    all_tiffnames.append(uv_aerosol_tiffnames)
    all_tiffnames.append(modis_tiffnames)
    all_tiffnames.append(era5_tiffnames)
    
    return all_tiffnames



def _parse_stats_mean(eopatch, product, path,day):
    # Tasks to calculate mean of data
    if product == 'NO2':
        eopatch.data['Daily'][eopatch.data['Daily'] == -9999.0] = np.nan
        eopatch.data['Daily'][eopatch.data['Daily'] > 1] = np.nan
                
    mean = MapFeatureTask((FeatureType.DATA,'Daily'),  # input features
                             (FeatureType.DATA_TIMELESS,'daily_product'),  # output feature
                             np.nanmean, # a function to apply to each feature
                             axis=0)
    # Task to export the merged tiff (mean of daily measurments)
    export_tiff = ExportToTiff((FeatureType.DATA_TIMELESS, 'daily_product'))

    ## Compute and export the mean daily value 
    mean(eopatch)
    tiffname = eopatch.meta_info['Names_'+ product][day]
    tiffname = tiffname.replace(tiffname[-6:], "00")
    export_tiff.execute(eopatch, filename=str(path)+'/'+product+'/'+tiffname)

def _parse_stats_mean_era5_u(eopatch, product, path,day):
    # Tasks to calculate mean of data
    mean_abs = MapFeatureTask((FeatureType.DATA,'Daily'),  # input features
                             (FeatureType.DATA_TIMELESS,'daily_product'),  # output feature
                             lambda f: np.nanmean(abs(f), axis=0)) # a function to apply to each feature
                             
    # Task to export the merged tiff (mean of daily measurments)
    export_tiff = ExportToTiff((FeatureType.DATA_TIMELESS, 'daily_product'))

    ## Compute and export the mean daily value 
    mean_abs(eopatch)
    tiffname = eopatch.meta_info['Names_'+ product][day]
    tiffname = tiffname.replace(tiffname[-10:], "00")
    export_tiff.execute(eopatch, filename=str(path)+'/'+product+'/'+tiffname)

def _parse_stats_mean_era5(eopatch, product, path,day):
    # Tasks to calculate mean of data
    mean_abs = MapFeatureTask((FeatureType.DATA,'Daily'),  # input features
                             (FeatureType.DATA_TIMELESS,'daily_product'),  # output feature
                             lambda f: np.nanmean(abs(f), axis=0)) # a function to apply to each feature
                             
    # Task to export the merged tiff (mean of daily measurments)
    export_tiff = ExportToTiff((FeatureType.DATA_TIMELESS, 'daily_product'))

    ## Compute and export the mean daily value 
    mean_abs(eopatch)
    tiffname = eopatch.meta_info['Names_'+ product][day]
    tiffname = tiffname.replace(tiffname[-6:], "00")
    export_tiff.execute(eopatch, filename=str(path)+'/'+product+'/'+tiffname)
    

def _parse_stats_sum(eopatch, product, path,day):
    # Tasks to calculate mean of data
    summation = MapFeatureTask((FeatureType.DATA,'Daily'),  # input features
                             (FeatureType.DATA_TIMELESS,'daily_product'), # output feature
                             lambda f: np.nansum(f, axis=0) # a function to apply to each feature
                             )
    
    # Task to export the merged tiff (mean of daily measurments)
    export_tiff = ExportToTiff((FeatureType.DATA_TIMELESS, 'daily_product'))

    ## Compute and export the mean daily value 
    summation(eopatch)
    tiffname = eopatch.meta_info['Names_'+ product][day]
    tiffname = tiffname.replace(tiffname[-6:], "00")
    export_tiff.execute(eopatch, filename=str(path)+'/'+product+'/'+tiffname)
            

def _parse_stats_max(eopatch, product, path,day):
    # Tasks to calculate maximum of data 
    maximum = MapFeatureTask((FeatureType.DATA,'Daily'),  # input features
                             (FeatureType.DATA_TIMELESS,'daily_product'),  # output feature
                             np.nanmax,                    # a function to apply to each feature
                             axis=0)
    
    # Task to export the merged tiff (mean of daily measurments)
    export_tiff = ExportToTiff((FeatureType.DATA_TIMELESS, 'daily_product'))
    
    ## Compute and export the maximum daily value 
    maximum(eopatch)
    tiffname = eopatch.meta_info['Names_'+ product][day]
    tiffname = tiffname.replace(tiffname[-6:], "00")
    export_tiff.execute(eopatch, filename=str(path)+'/'+product+'/'+tiffname)

def _parse_stats_min(eopatch, product, path,day):
    # Tasks to calculate minimum of data
    minimum = MapFeatureTask((FeatureType.DATA,'Daily'),  # input features
                             (FeatureType.DATA_TIMELESS,'daily_product'),  # output feature
                             np.nanmin,                    # a function to apply to each feature
                             axis=0)
    
    # Task to export the merged tiff (mean of daily measurments)
    export_tiff = ExportToTiff((FeatureType.DATA_TIMELESS, 'daily_product'))
    
    ## Compute and export the minimum daily value 
    minimum(eopatch)
    tiffname = eopatch.meta_info['Names_'+ product][day]
    tiffname = tiffname.replace(tiffname[-6:], "00")
    export_tiff.execute(eopatch, filename=str(path)+'/'+product+'/'+tiffname)


def _parse_stats_std(eopatch, product, path,day):
    # Tasks to calculate standard deviation of CAMS data
    std = MapFeatureTask((FeatureType.DATA,'Daily'),  # input features
                             (FeatureType.DATA_TIMELESS,'daily_product'),  # output feature
                             np.nanstd,                    # a function to apply to each feature
                             axis=0)
    
    # Task to export the merged tiff (mean of daily measurments)
    export_tiff = ExportToTiff((FeatureType.DATA_TIMELESS, 'daily_product'))
    
    ## Compute and export the minimum daily value 
    std(eopatch)
    tiffname = eopatch.meta_info['Names_'+ product][day]
    tiffname = tiffname.replace(tiffname[-6:], "00")
    export_tiff.execute(eopatch, filename=str(path)+'/'+product+'/'+tiffname)


STATS_PARSER = dict(mean= _parse_stats_mean,
                    mean_era5 = _parse_stats_mean_era5,
                    mean_era5_u = _parse_stats_mean_era5_u,
                    summation = _parse_stats_sum,
                        maximum= _parse_stats_max,
                        minimum= _parse_stats_min,
                        std= _parse_stats_std
                    )

def stats(eopatch, product,stats, out_path):
    """
    Compute daily statstics of eopatch (mean, maximum, minimum, and standard deviation)  
    
    :param eopatch: Eopatch where data are stored  
    :param product: Product name in the eopatch
    :param stats: Statitcs (mean, max, ...) 
    :param out_path: Path of directory where to save the new single daily observation
    
    :return: None
    """   
    assert stats in ['mean','mean_era5','mean_era5_u', 'maximum', 'minimum', 'std','summation']
    
    for name in eopatch.meta_info['Names_'+ product]:
        tiffname = name
        day, hour = name.split('_day')[-1].split('_')
        break
    
    if product == 'wind_u' and out_path.find('training_dataset_ita') == 0 and tiffname.find('.tif'):
        stats = 'mean_era5_u'
        
    # Task to put all measurements of same day together 
    add_daily_product = AddFeature((FeatureType.DATA, 'Daily'))

    ## extract day index of each tiff
    daily_index = []
    for name in eopatch.meta_info['Names_'+ product]:
        day, hour = name.split('_day')[-1].split('_')
        daily_index.append(day)
    # extract the indices of the same day    
    c =  defaultdict(list)
    for idx , day in enumerate(daily_index):
        c[day].append(idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for key,locs in c.items():
            all_hours_products = []
            for l in locs:
                hour_product = eopatch.data[product][l]
                all_hours_products.append(hour_product)    

            daily_product = np.stack(all_hours_products, axis=0)
            add_daily_product.execute(eopatch,daily_product)
            STATS_PARSER[stats](eopatch, product, out_path,day = l)

def stats_era5(eopatch: EOPatch, era5_products:list, out_path:str):
    """
    Helper function to compute ERA5 data statstics
    """
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
        for era5_p in era5_products:
            os.mkdir(out_path + '/' + era5_p)
                
    statis = 'mean_era5'
    for era5_p in era5_products:
        if era5_p == 'specific_rain_water_content':
            statis = 'summation'
            
        stats(eopatch=eopatch,
              product = era5_p,
             stats = statis,
             out_path= out_path)
        
def stats_cams(eopatch: EOPatch, cams_products:list, out_path:str):
    """
    Helper function to compute ERA5 data statstics
    """
    statistics = ['mean','maximum', 'minimum','std']
    for statis in statistics:
        ## creat dircetory for the daily statistcs
        stats_cams_dir = out_path + '/' + statis +'_cams'
        if not os.path.isdir(stats_cams_dir):
            os.makedirs(stats_cams_dir)
            for cams_p in cams_products:
                os.mkdir(stats_cams_dir + '/' + cams_p)
        ## Apply the stats function on CAMS products
        for cams_p in cams_products:
            stats(eopatch=eopatch,
                  product = cams_p,
                  stats = statis,
                  out_path= stats_cams_dir)

def resample_cams_to_target(target_bbox, AOI:str,model:str, in_path:str, out_path:str, tiffnames:list=None):
    """
    Resample CAMS data to the target resolution and the same bounding box of AOI  
    
    :param target_bbox: Bounding box of eopatch data (PM 2.5 , NO2)  
    :param AOI: Area of interest
    :param model: Model (PM2.5, NO2)
    :param in_path: Path of directory where CAMS data are stored
    :param out_path: Path of directory where to save the new resampled CAMS data
    :param tiffnames: list of CAMS images names
    
    :return: None
    """       
    assert AOI in ['Italy','California','South_Africa']
    assert model in ['NO2','PM2_5']

    if model == 'NO2':
        target_resolution = 1000
        t=1
    else:
        if AOI == 'Italy':
            target_resolution = 1000
            t = 1
        else:
            target_resolution = 10000
            t = 10

    target_size = bbox_to_dimensions(target_bbox, target_resolution)
    statistics = ['mean','maximum', 'minimum','std']
    for statis in statistics:
        ## Directory of stats CAMS
        stats_cams_dir = in_path +'/'+ statis +'_cams'
        cams_products = sorted(os.listdir(Path(stats_cams_dir)))
        ## Create direction for resampled CAMS
        resampled_cams_dir = f'{out_path}/resampled_{statis}_cams_{t}km'
        if not os.path.isdir(resampled_cams_dir):
            os.makedirs(resampled_cams_dir)
            for cams_p in cams_products:
                os.mkdir(resampled_cams_dir + '/' + cams_p) 

        if tiffnames != None:
            ## start resampling CAMS products - just target dates (in tiffnames)
            for cams_p in cams_products: # Resample (billinear)
                for tiff_name in tiffnames[0]:
                    for j in glob.glob(str(stats_cams_dir) + '/' + cams_p + '/' + tiff_name):
                        path_in = j
                        path_out = re.sub(str(stats_cams_dir), str(resampled_cams_dir), path_in)
                        upscale_tiff(path_in, path_out, target_size)

            for cams_p in cams_products: # Resample (billinear)
                for tiff_name in tiffnames[1]:
                    for j in glob.glob(str(stats_cams_dir) + '/' + cams_p + '/' + tiff_name):
                        path_in = j
                        path_out = re.sub(str(stats_cams_dir), str(resampled_cams_dir), path_in)
                        upscale_tiff(path_in, path_out, target_size)
        else:
            ## start resampling CAMS products
            for cams_p in cams_products: # Resample (billinear)
                for j in glob.glob(str(stats_cams_dir) + '/' + cams_p + '/*.tif'):
                    path_in = j
                    path_out = re.sub(str(stats_cams_dir), str(resampled_cams_dir), path_in)
                    upscale_tiff(path_in, path_out, target_size)


def resample_s5p_to_target(target_bbox, AOI:str,model:str, in_path:str, out_path:str, tiffnames:list=None):

    """
    Resample S5P data to the target resolution and the same bounding box of AOI  
    
    :param target_bbox: Bounding box of eopatch data (PM 2.5 , NO2)  
    :param AOI: Area of interest
    :param model: Model (PM2.5, NO2)
    :param in_path: Path of directory where CAMS data are stored
    :param out_path: Path of directory where to save the new resampled CAMS data
    :param tiffnames: list of S5P images names
    
    :return: None
    """  
    
    assert AOI in ['Italy','California','South_Africa']
    assert model in ['NO2','PM2_5']
    
    if model == 'NO2':
        target_resolution = 1000
        t=1
    else:
        if AOI == 'Italy':
            target_resolution = 1000
            t = 1
        else:
            target_resolution = 10000
            t = 10
            
    target_size = bbox_to_dimensions(target_bbox, target_resolution)
    s5p_products = sorted(os.listdir(Path(in_path)))
    
    out_path = f'{out_path}/resampled_s5p_{t}km'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
        for s5p_p in s5p_products:
            os.mkdir(out_path + '/' + s5p_p)
    if tiffnames != None:
        for index, s5p_p in enumerate(s5p_products): # Resample (billinear)
            for tiffname in tiffnames[index]:
                for j in glob.glob(str(in_path) + '/' + s5p_p + '/' + tiffname):
                    path_in = j
                    path_out = re.sub(str(in_path), str(out_path), path_in)
                    upscale_tiff(path_in, path_out, target_size)
    else:
        for s5p_p in s5p_products: # Resample (billinear)
            for j in glob.glob(str(in_path) + '/' + s5p_p + '/*.tif'):
                path_in = j
                path_out = re.sub(str(in_path), str(out_path), path_in)
                upscale_tiff(path_in, path_out, target_size)


def resample_era5_to_target(target_bbox, AOI:str,model:str, in_path:str, out_path:str, tiffnames:list=None):
    
    """
    Resample ERA5 data to the target resolution and the same bounding box of AOI  
    
    :param target_bbox: Bounding box of eopatch data (PM 2.5 , NO2)  
    :param AOI: Area of interest
    :param model: Model (PM2.5, NO2)
    :param in_path: Path of directory where CAMS data are stored
    :param out_path: Path of directory where to save the new resampled CAMS data
    :param tiffnames: list of ERA5 images names
    
    :return: None
    """  
    
    assert AOI in ['Italy','California','South_Africa']
    assert model in ['NO2','PM2_5']

    if model == 'NO2':
        target_resolution = 1000
        t=1
    else:
        if AOI == 'Italy':
            target_resolution = 1000
            t = 1
        else:
            target_resolution = 10000
            t = 10

    target_size = bbox_to_dimensions(target_bbox, target_resolution)

    ## Directory of merged ERA5
    mean_era5_dir = in_path
    era5_products = sorted(os.listdir(Path(mean_era5_dir)))

    ## Create direction for resampled ERA5
    out_path = f'{out_path}/resampled_era5_{t}km'
    resampled_era5_dir = out_path
    if not os.path.isdir(resampled_era5_dir):
        os.makedirs(resampled_era5_dir)
        for era5_p in era5_products:
            os.mkdir(resampled_era5_dir + '/' + era5_p) 
    if tiffnames != None:
        ## start resampling merged era5 - billinear
        for index , era5_p in enumerate(era5_products):
            for tiffname in tiffnames[index]:
                for j in glob.glob(str(mean_era5_dir) + '/' + era5_p + '/' + tiffname):
                    path_in = j
                    path_out = re.sub(str(mean_era5_dir), str(resampled_era5_dir), path_in)
                    upscale_tiff(path_in, path_out, target_size)
    else:   
        ## start resampling merged era5 - billinear
        for index , era5_p in enumerate(era5_products):
            for j in glob.glob(str(mean_era5_dir) + '/' + era5_p + '/*.tif'):
                path_in = j
                path_out = re.sub(str(mean_era5_dir), str(resampled_era5_dir), path_in)
                upscale_tiff(path_in, path_out, target_size)


def resample_modis_to_target(target_bbox, AOI:str, model:str, in_path:str, out_path:str, tiffnames:list=None):
    """
    Resample MODIS data to the target resolution and the same bounding box of AOI  
    
    :param target_bbox: Bounding box of eopatch data (PM 2.5 , NO2)  
    :param AOI: Area of interest
    :param model: Model (PM2.5, NO2)
    :param in_path: Path of directory where CAMS data are stored
    :param out_path: Path of directory where to save the new resampled CAMS data
    :param tiffnames: list of MODIS images names
    
    :return: None
    """  
    assert AOI in ['Italy','California','South_Africa']
    assert model in ['NO2','PM2_5']

    if model == 'NO2':
        target_resolution = 1000
        t=1
    else:
        if AOI == 'Italy':
            target_resolution = 1000
            t = 1
        else:
            target_resolution = 10000
            t = 10

    target_size = bbox_to_dimensions(target_bbox, target_resolution)

    ## Directory of merged ERA5
    modis_train_dir = in_path
    modis_products = sorted(os.listdir(Path(modis_train_dir)))

    ## Create direction for resampled ERA5
    out_path = f'{out_path}/resampled_modis_{t}km'
    resampled_modis_dir = out_path
    if not os.path.isdir(resampled_modis_dir):
        os.makedirs(resampled_modis_dir)
        for modis_p in modis_products:
            os.mkdir(resampled_modis_dir + '/' + modis_p) 

    if tiffnames != None:
        ## Resampling MODIS - mean
        for modis_p in modis_products:
            for tiffname in tiffnames:
                for j in glob.glob(str(modis_train_dir) + '/' + modis_p + '/' + tiffname):
                    path_in = j
                    path_out = re.sub(str(modis_train_dir), str(resampled_modis_dir), path_in)
                    upscale_mean_tiff(path_in, path_out, target_size)
    else:
        ## Resampling MODIS - mean
        for modis_p in modis_products:
            for j in glob.glob(str(modis_train_dir) + '/' + modis_p + '/*.tif'):
                path_in = j
                path_out = re.sub(str(modis_train_dir), str(resampled_modis_dir), path_in)
                upscale_mean_tiff(path_in, path_out, target_size)   

                
def _parse_extract_modis(points,path,label):
    #print(str(path) + '/'+ label +'/*.tif')
    modis_paths = glob.glob(str(path) + '/'+ label +'/*.tif')
    modis_df = extract_points_from_modis(points, modis_paths, label)
    modis_df = pd.concat(modis_df)
    modis_df = modis_df.dropna()
    return modis_df

def _parse_extract_era5(points,path,label,column1,column2):
    era5_paths = glob.glob(str(path) + '/'+ label +'/*.tif')
    era5_df = extract_points_from_era5(points, era5_paths, column1, column2)
    era5_df = pd.concat(era5_df)
    era5_df = era5_df.dropna()
    return era5_df

def _parse_extract_lc(points, path, column):
    lc_df = extract_points_from_timeless_raster(points, path, column)
    lc_df = pd.concat(lc_df)
    lc_df = lc_df.dropna()
    return lc_df



def _parse_extract_cams(points, path, label, column):
    cams_paths = glob.glob(str(path) + '/'+ label +'/*.tif')
    cams_df = extract_points_from_raster(points, cams_paths, column)
    cams_df = pd.concat(cams_df)
    cams_df = cams_df.dropna()
    return cams_df

def _parse_extract_s5p(points, path, label, column):
    s5p_paths = glob.glob(str(path) + '/'+ label +'/*.tif')
    s5p_df = extract_points_from_raster(points, s5p_paths, column)
    s5p_df = pd.concat(s5p_df)
    s5p_df = s5p_df.dropna()
    return s5p_df

EXTRACT_PARSER = dict(modis= _parse_extract_modis,
                      era5 = _parse_extract_era5,
                      lc = _parse_extract_lc,
                      cams = _parse_extract_cams,
                      s5p = _parse_extract_s5p
                      )

def extract_data(predictor:str, points, model:str,AOI:str, path:str,
                 label:str=None,predictor_bbox=None, target_eop=None,feature_name=None):
    """
    Sampling points from MODIS scenes 
    :param points_gdf: Geodataframe of sampling points 
    :param raster_paths: Directory file where MODIS images are stored
    :param name: List of modis images name
    
    :return: Dataframe of extracted points values and coordinates
    """ 
    assert AOI in ['Italy','California','South_Africa']
    assert model in ['NO2','PM2_5']
    assert predictor in ['modis', 'era5', 'lc', 'cams','s5p']
    
    if model == 'NO2':
        target_resolution = 1000
        t=1
    else:
        if AOI == 'Italy':
            target_resolution = 1000
            t = 1
        else:
            target_resolution = 10000
            t = 10

    if predictor == 'cams':
        bounds = list(predictor_bbox)
        xres = (bounds[2]-bounds[0])/target_eop.data[model].shape[1]
        yres = (bounds[3]-bounds[1])/target_eop.data[model].shape[2]
        bounds = geometry.box(bounds[0], bounds[1], bounds[2], bounds[3])
        bounds = gpd.GeoDataFrame(geometry=[bounds], crs="EPSG:4326")
        points = gpd.overlay(points, bounds, how='intersection')
        
        cams_df = []
        statistics = ['mean','maximum','minimum','std']
        for stats in statistics:
            cams_path = path+f'/resampled_{stats}_cams_{t}km'
            df = EXTRACT_PARSER[predictor](points, path=cams_path, label=label, column= f'{stats}_cams_{label}')
            cams_df.append(df)
        return cams_df

    elif predictor == 's5p':
        bounds = list(predictor_bbox)
        xres = (bounds[2]-bounds[0])/target_eop.data[model].shape[1]
        yres = (bounds[3]-bounds[1])/target_eop.data[model].shape[2]
        bounds = geometry.box(bounds[0], bounds[1], bounds[2], bounds[3])
        bounds = gpd.GeoDataFrame(geometry=[bounds], crs="EPSG:4326")
        points = gpd.overlay(points, bounds, how='intersection')
        
        s5p_path = path+f'/resampled_s5p_{t}km'
        df = EXTRACT_PARSER[predictor](points, s5p_path, label=label, column= f's5p_{label}')
        return df

    elif predictor == 'era5':
        bounds = list(predictor_bbox)
        xres = (bounds[2]-bounds[0])/target_eop.data[model].shape[1]
        yres = (bounds[3]-bounds[1])/target_eop.data[model].shape[2]
        bounds = geometry.box(bounds[0], bounds[1], bounds[2], bounds[3])
        bounds = gpd.GeoDataFrame(geometry=[bounds], crs="EPSG:4326")
        points = gpd.overlay(points, bounds, how='intersection')

        era5_path = path+f'/resampled_era5_{t}km'
        era5_df = []
        era5_products = [('relative_humidity','p_rh'),('specific_rain_water_content','p_srwc'),
                         ('wind_u','p_wind_u'),('wind_v','p_wind_v')]
        for era5, previous_era5 in era5_products:
            df = EXTRACT_PARSER[predictor](points, path=era5_path, label=era5,
                                           column1= era5, column2=previous_era5)
            era5_df.append(df)
        return era5_df

    elif predictor == 'modis':
        bounds = list(predictor_bbox)
        xres = (bounds[2]-bounds[0])/target_eop.data[model].shape[1]
        yres = (bounds[3]-bounds[1])/target_eop.data[model].shape[2]
        bounds = geometry.box(bounds[0], bounds[1], bounds[2], bounds[3])
        bounds = gpd.GeoDataFrame(geometry=[bounds], crs="EPSG:4326")
        points = gpd.overlay(points, bounds, how='intersection')
        
        modis_df = []
        modis_path = path+f'/resampled_modis_{t}km'
        columns = ['daily_mean_AOD', 'daily_minimum_AOD', 'daily_maximum_AOD']
        for column in columns:
            df = EXTRACT_PARSER[predictor](points, path= modis_path, label= column)
            modis_df.append(df)
        modis_df[0].rename(columns={"daily_mean_AOD": "MODIS_mean"})
        modis_df[1].rename(columns={"daily_minimum_AOD": "MODIS_min"})
        modis_df[2].rename(columns={"daily_maximum_AOD": "MODIS_max"})
        return modis_df

    elif predictor == 'lc':
        if AOI == 'California':
            lc_df = []
            land_cover = ['agri','natural','urban','water']
            for lc in land_cover:
                lc_path = [path+f'/{label}_{model}_{lc}_{t}km.tif']
                df = EXTRACT_PARSER[predictor](points, path=lc_path, column=f'{lc}_{t}km')
                lc_df.append(df)
                lc_path = [path+f'/{label}_{model}_{lc}_50km.tif']
                df = EXTRACT_PARSER[predictor](points, path=lc_path, column=f'{lc}_50km')
                lc_df.append(df)
        if AOI == 'South_Africa':
            lc_df = []
            land_cover = ['agri','natural','urban','indtrans']
            for lc in land_cover:
                lc_path = [path+f'/{label}_{model}_{lc}_{t}km.tif']
                df = EXTRACT_PARSER[predictor](points, path=lc_path, column=f'{lc}_{t}km')
                lc_df.append(df)
                lc_path = [path+f'/{label}_{model}_{lc}_50km.tif']
                df = EXTRACT_PARSER[predictor](points, path=lc_path, column=f'{lc}_50km')
                lc_df.append(df)
        if AOI == 'Italy':
            lc_df = []
            land_cover = ['agri','natural','urban','indtrans','water']
            for lc in land_cover:
                lc_path = [path+f'/{label}_{model}_{lc}_{t}km.tif']
                df = EXTRACT_PARSER[predictor](points, path=lc_path, column=f'{lc}_{t}km')
                lc_df.append(df)
                lc_path = [path+f'/{label}_{model}_{lc}_50km.tif']
                df = EXTRACT_PARSER[predictor](points, path=lc_path, column=f'{lc}_50km')
                lc_df.append(df)
        return(lc_df)


def extract_points_from_modis(points_gdf, raster_paths, name):
    """
    Helper function to extract points from MODIS scenes 
    :param points_gdf: Geodataframe of sampling points 
    :param raster_paths: Directory file where MODIS images are stored
    :param name: List of modis images name
    
    :return: Dataframe of extracted points values and coordinates
    """ 
    
    rast = rio.open(raster_paths[0])    
    x = []
    y = []
    row = []
    col = []
    for point in points_gdf['geometry']:
        row2, col2 = rast.index(point.xy[0][0],point.xy[1][0])
        x.append(point.xy[0][0])
        y.append(point.xy[1][0])
        row.append(row2)
        col.append(col2)
        
    res = []
    for raster_path in raster_paths:
        rast2 = rio.open(raster_path)
        day = raster_path.split('_day')[-1]
        day = day.split('_')[0].split('.')[0]
        res2 = pd.DataFrame({'lon':x, 'lat':y, name:rast2.read(1)[row,col], 'date':int(day)})
        res.append(res2)
        
    return res

def extract_points_from_raster(points_gdf, raster_paths, name):
    """
    Helper function to extract points from any raster     
    :param points_gdf: Geodataframe of sampling points 
    :param raster_paths: Directory file where rasters are stored
    :param name: List of rasters name
    
    :return: Dataframe of extracted points values and coordinates
    """    
    rast = rio.open(raster_paths[0])    
    x = []
    y = []
    row = []
    col = []
    for point in points_gdf['geometry']:
        row2, col2 = rast.index(point.xy[0][0],point.xy[1][0])
        x.append(point.xy[0][0])
        y.append(point.xy[1][0])
        row.append(row2)
        col.append(col2)
        
    res = []
    for raster_path in raster_paths:
        rast2 = rio.open(raster_path)
        day, hour = raster_path.split('_day')[-1].split('_')
        res2 = pd.DataFrame({'lon':x, 'lat':y, name:rast2.read(1)[row,col], 'date':int(day),'hour':int(hour[1:3])})
        res.append(res2)
        
    return res

def extract_points_from_era5(points_gdf, raster_paths, name, name2):
    """
    Helper function to extract points from ERA5 scenes     
    :param points_gdf: Geodataframe of sampling points 
    :param raster_paths: Directory file where ERA5 images are stored
    :param name: List of ERA5 images name
    
    :return: Dataframe of extracted points values and coordinates
    """    
    rast = rio.open(raster_paths[0])    
    x = []
    y = []
    row = []
    col = []
    for point in points_gdf['geometry']:
        row2, col2 = rast.index(point.xy[0][0],point.xy[1][0])
        x.append(point.xy[0][0])
        y.append(point.xy[1][0])
        row.append(row2)
        col.append(col2)
        
    res = []
    day, hour = raster_paths[0].split('_day')[-1].split('_')
    
    ## this to fill the obsrevation of previous day of first image with nan for training data,
    ## and with same values of first day for valdation and test data
    if str(raster_paths).find("train") == 0:
        res2 = pd.DataFrame({'lon':x, 'lat':y, name:rast.read(1)[row,col],name2:np.nan, 'date':int(day),'hour':int(hour[1:3])})
    else:
        res2 = pd.DataFrame({'lon':x, 'lat':y, name:rast.read(1)[row,col],name2:rast.read(1)[row,col], 'date':int(day),'hour':int(hour[1:3])})
        
    res.append(res2)
    p_values = rast.read(1)[row,col]
    
    for raster_path in raster_paths[1:]:
        rast2 = rio.open(raster_path)
        day, hour = raster_path.split('_day')[-1].split('_')
        res2 = pd.DataFrame({'lon':x, 'lat':y, name:rast2.read(1)[row,col],name2:p_values ,'date':int(day),'hour':int(hour[1:3])})
        res.append(res2)
        p_values = rast2.read(1)[row,col]           
    return res

    
def extract_points_from_timeless_raster(points_gdf, raster_paths, name):
    """
    Helper function to extract points from timeless raster like landcover     
    :param points_gdf: Geodataframe of sampling points 
    :param raster_paths: Directory file where timeless images are stored
    :param name: List of images name
    
    :return: Dataframe of extracted points values and coordinates
    """ 
    rast = rio.open(raster_paths[0])    
    x = []
    y = []
    row = []
    col = []
    for point in points_gdf['geometry']:
        row2, col2 = rast.index(point.xy[0][0],point.xy[1][0])
        x.append(point.xy[0][0])
        y.append(point.xy[1][0])
        row.append(row2)
        col.append(col2)
        
    res = []
    for raster_path in raster_paths:
        rast2 = rio.open(raster_path)
        res2 = pd.DataFrame({'lon':x, 'lat':y, name:rast2.read(1)[row,col]})
        res.append(res2)
        
    return res        


def masking_tiff(geo_filename: str, input_filename: str, output_filename: str):
    """ Mask a tiff file given a polygon geometry
    
    Taken from https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html?highlight=crop#masking-a-raster-using-a-shapefile
    """
    with fiona.open(geo_filename, 'r') as shapefile:
        shapes = [feature['geometry'] for feature in shapefile]
        
    with rasterio.open(input_filename) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, 
                                                      #crop=True, 
                                                      nodata=np.nan)
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1], # x
                     "width": out_image.shape[2],  # y
                     "transform": out_transform})


    with rasterio.open(output_filename, 'w', **out_meta) as dest:
        dest.write(out_image)