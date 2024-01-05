import geemap
import pandas as pd
import numpy as np
import os

#Earth Engine
import ee

print('Things imported')

#Authentication
ee.Authenticate()
ee.Initialize(project = 'ee-sfgiantsmatt')

print('Authenticated')

#Reading in data created in separate notebook
data = pd.read_csv('merged_df.csv')
data = data.set_index('GEOID')
data = data[data.get('ALAND_SQMI') > 0]

#Total images used, training + validation
batch_size = 1000

#Fraction of images used for training
train_frac = 0.8
tracts_sample = data.sample(batch_size)

#Dictionary used in filepath
train_or_val = {True: 'training/images',
                False: 'validation/images'}

#Creating dataframes with labels for the neural network
training_df = pd.DataFrame(columns = ['GEOID', 'Density'])
validation_df = pd.DataFrame(columns = ['GEOID', 'Density'])

#To track progress
counter = 0

print("Starting loop")

#Looping through every census tract in the sample
for id in tracts_sample.get('Str_GEOID').to_numpy():

    #Used to assign tract to either the training or validation set
    rand = np.random.rand() < train_frac

    int_id = int(id)

    #Somewhat arbitrarily defining the region of interest based on lat/long
    lat_long_range = 0.005
    roi = ee.Geometry.Rectangle([[data.get('INTPTLONG').loc[int_id] - lat_long_range, \
                                data.get('INTPTLAT').loc[int_id] - lat_long_range], \
                                [data.get('INTPTLONG').loc[int_id] + lat_long_range, \
                                data.get('INTPTLAT').loc[int_id] + lat_long_range]])

    #Getting the satellite imagery with low cloud coverage
    #Not strict about when the imagery is from, hopefully it doesn't matter too much
    collection = (ee.ImageCollection('COPERNICUS/S2')
                .filterBounds(roi)
                .filterDate('2010-01-01', '2021-12-31')
                .sort('CLOUDY_PIXEL_PERCENTAGE'))

    #Gets the image with the least cloud coverage
    image = ee.Image(collection.first())

    #Defines the output path and file name
    out_img = os.path.expanduser(f"~/Desktop/coding stuff/satelliteimageryproject/data/{train_or_val[rand]}/{data.get('Str_GEOID').loc[int_id]}.png")
    
    #Stole from various tutorials, not quite sure what each one does
    #Turns out the bands are red, blue, and green
    vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1}

    #Sets the image to only the region of interest (I think)
    image = image.clip(roi)

    #Downloads the image thumbnail
    geemap.get_image_thumbnail(image, out_img, vis_params, dimensions= (300, 300), format='png')

    #Puts the data for this tract into the correct csv file
    if rand:
        training_df.loc[len(training_df)] = [data.get('Str_GEOID').loc[int_id], data.get('Density').loc[int_id]]
    else:
        validation_df.loc[len(validation_df)] = [data.get('Str_GEOID').loc[int_id], data.get('Density').loc[int_id]]
    
    #Iterates counter and displays progress
    counter += 1
    print(f'Images downloaded:{counter}/{batch_size}')

#Downloads the labeled csv's    
training_df.to_csv(path_or_buf= os.path.expanduser("~/Desktop/coding stuff/satelliteimageryproject/data/training/training_labels.csv"))
validation_df.to_csv(path_or_buf= os.path.expanduser("~/Desktop/coding stuff/satelliteimageryproject/data/validation/validation_labels.csv"))
    