# The AI4EO Air Quality & Health challenge
The challenge is to downscale air quality products to a resolution that can be used on a local level. The focus is on the pollutants PM2.5 and NO2, which are amongst the most important to consider for health effects.


## Proposed Solutions

Air quality data downscaled by Random Forest model between in-situ data and input data (NO2/PM2.5) and auxiliary data resampled to target resolution. The solution takes advantage of spatial variability of land cover and temporal variability of metrological data, by computing the percentage of land cover classes inside each pixel and the surrounded area and compute the statistics of ERA5 products for the same and previous days. In addition to daily statistics of other predictors like CAMS and MODIS. 

The model calibrated by two steps, the first step was residual error correction by aggregate the predicted pixels that constitute the low-resolution pixel in the mean daily input (NO2/PM2.5), then compute the difference between them and distribute the difference equally between the predicted pixels. The second step was calibrating the corrected predictions with hourly NO2/PM2.5 data, by make the constitute corrected daily predictions and hourly low-resolution pixels to have the same averages.


## Results
Consisting of downscaled CAMS PM2.5 products, from their original spatial resolution of 10x10 km2 to 1x1 km2 over Europe (from 40x40 km2 to 10x10 km2 outside Europe); and Sentinel-5P Level-2 NO2 products, from their original spatial resolution of 7x3.5 km2 to 1x1 km2.


## Examples
The example of applying the model can be found [here](https://github.com/masawdah/air_quality/blob/master/examples)
