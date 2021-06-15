# Built-in modules
import os
from os import listdir as list_dir
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


## This function to merge data based on same date & location
def create_dataset(target_points, model:str, AOI:str, dataset:list, dates):
    assert model in ['NO2', 'PM2_5']
    assert AOI in ['Italy', 'California','South_Africa']
    if dates is not None: ## This is for validation dataset
        if model == 'NO2':
            pred_df = dataset[8]
            pred_df = pred_df[pred_df['date'].isin(list(dates['date']))]
            pred_df = pd.merge(pred_df, dates, how="left", on=["date"])
            
            ## Inner join by time & coordinates (CAMS - PM 2.5)
            pred_df = pd.merge(pred_df, dataset[0], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[1], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[2], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[3], on=['date','hour','lon', 'lat'], how='inner')

            ## Inner join by time & coordinates (CAMS - NO2 surface)
            pred_df = pd.merge(pred_df, dataset[4], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df,dataset[5], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[6], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[7], on=['date','hour','lon', 'lat'], how='inner')
            
            
            if AOI == 'Italy' or AOI == 'California':
                ## Inner join by time & coordinates (S5P - NO2 & UV Aerosol Index)
                pred_df = pd.merge(pred_df, dataset[9], on=['date','hour','lon', 'lat'], how='inner')


            ## Inner join by time & coordinates (ERA5)
            pred_df = pd.merge(pred_df, dataset[10], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[11], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[12], on=['date', 'hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[13], on=['date','hour','lon', 'lat'], how='inner')

            ## Inner join by time & coordinates (MODIS)
            pred_df = pd.merge(pred_df, dataset[14], on=['date','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[15], on=['date','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[16], on=['date','lon', 'lat'], how='inner')

            ## Inner join by coordinates (Land cover)
            pred_df = pd.merge(pred_df, dataset[17], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[18], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[19], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[20], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[21], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[22], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[23], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[24], on=['lon', 'lat'], how='inner')
            
            if AOI == 'Italy':
                pred_df = pd.merge(pred_df, dataset[25], on=['lon', 'lat'], how='inner')
                pred_df = pd.merge(pred_df, dataset[26], on=['lon', 'lat'], how='inner')

            
            return pred_df
                    

        elif model == 'PM2_5':
            pred_df = dataset[0]
            pred_df = pred_df[pred_df['date'].isin(list(dates['date']))]
            pred_df = pd.merge(pred_df, dates, how="left", on=["date"])

            ## Inner join by time & coordinates (CAMS - PM 2.5)
            pred_df = pd.merge(pred_df, dataset[1], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[2], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[3], on=['date','hour','lon', 'lat'], how='inner')

            ## Inner join by time & coordinates (CAMS - NO2 surface)
            pred_df = pd.merge(pred_df, dataset[4], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[5], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[6], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[7], on=['date','hour','lon', 'lat'], how='inner')
            
            
            if AOI == 'Italy' or AOI == 'California':
                ## Inner join by time & coordinates (S5P - NO2 & UV Aerosol Index)
                pred_df = pd.merge(pred_df, dataset[8], on=['date','hour','lon', 'lat'], how='inner')
                pred_df = pd.merge(pred_df, dataset[9], on=['date','hour','lon', 'lat'], how='inner')

            ## Inner join by time & coordinates (ERA5)
            pred_df = pd.merge(pred_df, dataset[10], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[11], on=['date','hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[12], on=['date', 'hour','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[13], on=['date','hour','lon', 'lat'], how='inner')

            ## Inner join by time & coordinates (MODIS)
            pred_df = pd.merge(pred_df, dataset[14], on=['date','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[15], on=['date','lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[16], on=['date','lon', 'lat'], how='inner')

            ## Inner join by coordinates (Land cover)
            pred_df = pd.merge(pred_df, dataset[17], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[18], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[19], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[20], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[21], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[22], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[23], on=['lon', 'lat'], how='inner')
            pred_df = pd.merge(pred_df, dataset[24], on=['lon', 'lat'], how='inner')
            
            if AOI == 'Italy':
                pred_df = pd.merge(pred_df, dataset[25], on=['lon', 'lat'], how='inner')
                pred_df = pd.merge(pred_df, dataset[26], on=['lon', 'lat'], how='inner')
            
            
            return pred_df
            


    else: ## This is for training dataset
        if AOI == 'California' and model == 'PM2_5' : # There are stations at the same location for California - here we calculated the mean value of those stations
            pm25_gt = target_points
            pm25g_gt_mean = pm25_gt.groupby(['SITE_LONGI','SITE_LATIT','Date']).mean()
            pm25g_gt_mean= pm25g_gt_mean.rename(columns={"AirQuality": "avg_AirQuality"})
            pm25_gt = pd.merge(pm25_gt, pm25g_gt_mean, on=['SITE_LONGI','SITE_LATIT','Date'], how='inner')
            pm25_gt= pm25_gt.drop(columns=['AirQuality'])
            pm25_gt= pm25_gt.rename(columns={"avg_AirQuality": "AirQuality"})
            pm25_gt = pm25_gt.drop_duplicates()
            target_points = pm25_gt
            
        # Rround corrdinates of stations to 6 digits  - to be the same as the coordinates of extracted data
        target_points= target_points.rename(columns={"SITE_LATIT": "lat", "SITE_LONGI": "lon"})
        for i, point in enumerate(target_points['geometry']):
            target_points.loc[i,'lon'] = round(point.xy[0][0],6)
            target_points.loc[i,'lat'] = round(point.xy[1][0],6)
            
        target_points['date'] = target_points['Date']
        target_points = target_points.dropna()

        if model == 'PM2_5':
            ## Inner join by time & coordinates (CAMS - PM 2.5)
            train_df = pd.merge(target_points, dataset[0], on=['date','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[1], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[2], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[3], on=['date','hour','lon', 'lat'], how='inner')

            ## Inner join by time & coordinates (CAMS - NO2 surface)
            train_df = pd.merge(train_df, dataset[4], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[5], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[6], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[7], on=['date','hour','lon', 'lat'], how='inner')

            
            if AOI == 'Italy' or AOI == 'California':
                ## Inner join by time & coordinates (S5P - NO2 & UV Aerosol Index)
                train_df = pd.merge(train_df, dataset[8], on=['date','hour','lon', 'lat'], how='inner')
                train_df = pd.merge(train_df, dataset[9], on=['date','hour','lon', 'lat'], how='inner')



            ## Inner join by time & coordinates (ERA5)
            train_df = pd.merge(train_df, dataset[10], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[11], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[12], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[13], on=['date','hour','lon', 'lat'], how='inner')

            ## Inner join by time & coordinates (MODIS)
            train_df = pd.merge(train_df, dataset[14], on=['date','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[15], on=['date','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[16], on=['date','lon', 'lat'], how='inner')

            ## Inner join by coordinates (Land cover)
            train_df = pd.merge(train_df, dataset[17], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[18], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[19], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[20], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[21], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[22], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[23], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[24], on=['lon', 'lat'], how='inner')
            
            if AOI == 'Italy':
                train_df = pd.merge(train_df, dataset[25], on=['lon', 'lat'], how='inner')
                train_df = pd.merge(train_df, dataset[26], on=['lon', 'lat'], how='inner')


            return train_df
                

        elif model == 'NO2':
            ## Inner join by time & coordinates (S5P - NO2)
            train_df = pd.merge(target_points, dataset[8], on=['date','lon', 'lat'], how='inner')
            
            ## Inner join by time & coordinates (CAMS - PM 2.5)
            train_df = pd.merge(train_df, dataset[0], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[1], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[2], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[3], on=['date','hour','lon', 'lat'], how='inner')

            ## Inner join by time & coordinates (CAMS - NO2 surface)
            train_df = pd.merge(train_df, dataset[4], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[5], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[6], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[7], on=['date','hour','lon', 'lat'], how='inner')

            if AOI == 'Italy' or AOI == 'California':
                ## Inner join by time & coordinates (S5P - UV Aerosol Index)
                train_df = pd.merge(train_df, dataset[9], on=['date','hour','lon', 'lat'], how='inner')


            ## Inner join by time & coordinates (ERA5)
            train_df = pd.merge(train_df, dataset[10], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[11], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[12], on=['date','hour','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[13], on=['date','hour','lon', 'lat'], how='inner')

            ## Inner join by time & coordinates (MODIS)
            train_df = pd.merge(train_df, dataset[14], on=['date','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[15], on=['date','lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[16], on=['date','lon', 'lat'], how='inner')

            ## Inner join by coordinates (Land cover)
            train_df = pd.merge(train_df, dataset[17], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[18], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[19], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[20], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[21], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[22], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[23], on=['lon', 'lat'], how='inner')
            train_df = pd.merge(train_df, dataset[24], on=['lon', 'lat'], how='inner')

            if AOI == 'Italy':
                train_df = pd.merge(train_df, dataset[25], on=['lon', 'lat'], how='inner')
                train_df = pd.merge(train_df, dataset[26], on=['lon', 'lat'], how='inner')

                
            ## convert the unit of stations to the same unit of s5p NO2
            train_df['AirQuality'] = (train_df['AirQuality'] / (1.0e-15 * 1.9125))/(6.02214e+19)
            
            return train_df
            
            

        
