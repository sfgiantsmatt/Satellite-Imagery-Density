import pandas as pd
import numpy as np
import tensorflow as tf
import keras as keras

# Architecture copied from ChatGPT
model = tf.keras.Sequential([
    #tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    #tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Not very familiar with any other optimizers, not really sure what they do
# Mean squared error used because the prediction goal is continuous
# Mean absolute error used as a metric because the number is smaller which makes me feel better
model.compile(
    optimizer= 'Adam', 
    loss='mean_squared_error', 
    metrics=['mae'])

# Copied this from ChatGPT, although I took out a line that added random scaling since the scale is important (I think)
# Shears and horizontal flips shouldn't change much, so I kept them in
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values to the range [0, 1]
    shear_range=0.2,        # Shear transformations
    horizontal_flip=True    # Random horizontal flip
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# CSV's that I automatically generated after the image downloading concluded
# GEOID column (FIPS code I think), has code for each census tract image (integer, probably should've saved them as strings)
# Density column that has population density for each census tract that was randomly selected, calculated from Gazetteer file and P1 file
train_df = pd.read_csv('data/training/training_labels.csv')
val_df = pd.read_csv('data/validation/validation_labels.csv')

# Changing the type of the GEOID column and adding .png, probably could've done this at the time of downloading if I knew it was gonna be necessary
train_df = train_df.assign(GEOID = train_df.get('GEOID').apply(lambda x: str(x) + '.png'))
val_df = val_df.assign(GEOID = val_df.get('GEOID').apply(lambda x: str(x) + '.png'))

# 32 batch size is pretty arbitrary I think, I didn't change what ChatGPT gave me
# Image size of 300 by 300 was specified at time of download
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    'data/training/images',
    x_col = 'GEOID',
    y_col = 'Density',
    target_size=(300, 300),  
    batch_size= 32,
    class_mode='raw'       # Regression
)

validation_generator = test_datagen.flow_from_dataframe(
    val_df,
    'data/validation/images',
    x_col = 'GEOID',
    y_col = 'Density',
    target_size=(300, 300),
    batch_size= 32,
    class_mode='raw'
)

# Doesn't seem to improve after 15 epochs (went up to 30 one time, took a while for my computer lol)
# Copied from ChatGPT (except epoch number)
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs= 15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Best mean absolute error achieved in a particular epoch: about 4200.
# Probably not great to be honest but it starts around 7000.