import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from typing import List, Union, Tuple, Optional
import pandas as pd

def polygrid(poly, res, crs):
    """"Function to create a square polygon grid with a given resolution from a polygon bounding box"""
    # Get corners
    xmin,ymin,xmax,ymax = poly.total_bounds
    # Define poly vertical and horizontal limits
    cols = list(np.arange(xmin, xmax, res))
    rows = list(np.arange(ymin, ymax, res))
    rows.reverse()
    # Create polygons
    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(Polygon([(x,y), (x+res, y), (x+res, y+res), (x, y+res)]))
    # Return gpd
    grid = gpd.GeoDataFrame({'geometry':polygons}, crs=crs)
    return grid

def recpolygrid(poly, xres,yres, crs):
    """"Function to create a square polygon grid with a given resolution from a polygon bounding box"""
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

def percs_landuse(zstats, urban, indtrans, agri, natural):   
    for indx, vals in enumerate(zstats):
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
    return urban, indtrans, agri, natural


def export_geotiff(path, rast, trans, epsg):
    new_dataset = rio.open(path, 'w', driver='GTiff',
                           height = rast.shape[0], width = rast.shape[1],
                           count=1, dtype=str(rast.dtype),
                           crs="EPSG:"+str(epsg),
                           transform=trans)
    new_dataset.write(rast, 1)
    new_dataset.close()
    
def upscale_mean_tiff(input_filename: str, output_filename: str, out_shape: Tuple[int, int]):
    """ Upscale a tiff file given a target shape. Writes out first channel only """
    
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
        

def extract_points_from_raster(points_gdf, raster_paths, name):
    """  """    
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
        res2 = pd.DataFrame({'lon':x, 'lat':y, name:rast2.read(1)[row,col], 'raster':raster_path})
        res.append(res2)
        
    return res


