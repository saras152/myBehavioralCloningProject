# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 08:56:03 2017

@author: raghunath
"""

import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda, Cropping2D,Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D


print('building the model')
model=Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5 , input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((80,20),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='same',activation='relu')) 
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='same',activation='relu')) 
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='same',activation='relu')) 
#, W_regularizer=l2(0.001)
model.add(Convolution2D(24, 3, 3, border_mode='same',activation='relu')) 
model.add(Dropout(0.5))
model.add(Convolution2D(24, 3, 3, border_mode='same',activation='relu')) 
model.add(Dropout(0.5))
model.add(Convolution2D(24, 3, 3, border_mode='same',activation='relu')) 
model.add(Dropout(0.5))
model.add(Flatten()) 
model.add(Dense(80,activation='relu')) 
model.add(Dense(50,activation='relu')) 
model.add(Dense(30,activation='relu')) 
model.add(Dense(10,activation='relu')) 
model.add(Dense(1)) 
model.compile(optimizer='adam', loss='mse') 

model.summary()

os.chdir("E:\\RaNa_E\\ocv_work\\Project3\\14sept2017")

images=[]
measurements=[]

def myXYtranslate(image,Angle,XYrange):
    rows, cols, _ = image.shape
    Xchange = XYrange*np.random.uniform()-XYrange/2
    AngleXY = Angle + Xchange/XYrange
    Ychange = (XYrange*np.random.uniform()-XYrange/2)/3
    Matr = np.float32([[1,0,Xchange],[0,1,Ychange]])
    imgXY = cv2.warpAffine(image,Matr,(cols,rows))
    return imgXY,AngleXY

print('Reading the straight driven files')
lines=[]
csvfile=open('driving_log.csv')
reader=csv.reader(csvfile)
for line in reader:
    lines.append(line)
    #['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    

for line in lines:
    if abs(float(line[3]))>=0.00000:
        
        image=cv2.imread(line[0])#center image picked
        #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #image=image[:,:,np.newaxis]
        images.append(image)
        measurement=float(line[3])#steering angle picked
        measurements.append(measurement)
        image_flipped = np.fliplr(image)
        images.append(image_flipped)
        measurement_flipped = -measurement
        measurements.append(measurement_flipped)
        
            
        image=cv2.imread(line[1].strip())#left image picked
        #print('E:\\RaNa_E\\ocv_work\\Project3\\data\\data\\'+line[1])
        #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #image=image[:,:,np.newaxis]
        images.append(image)
        measurement=float(line[3])+0.2#steering angle picked
        measurements.append(measurement)
        image_flipped = np.fliplr(image)
        images.append(image_flipped)
        measurement_flipped = -measurement
        measurements.append(measurement_flipped)
        
        image=cv2.imread(line[2].strip())#right image picked
        #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #image=image[:,:,np.newaxis]
        images.append(image)
        measurement=float(line[3])-0.2#steering angle picked
        measurements.append(measurement)
        image_flipped = np.fliplr(image)
        images.append(image_flipped)
        measurement_flipped = -measurement
        measurements.append(measurement_flipped)
        
        if abs(float(line[3]))>0.2:
            image=cv2.imread(line[0].strip())#center image picked
            image=cv2.GaussianBlur(image,(3,3),0)
            #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #image=image[:,:,np.newaxis]
            images.append(image)
            measurement=float(line[3])#steering angle picked
            measurements.append(measurement)
            image_flipped = np.fliplr(image)
            images.append(image_flipped)
            measurement_flipped = -measurement
            measurements.append(measurement_flipped)
            
        if abs(float(line[3]))>0.4:
            image=cv2.imread(line[0])#center image picked
            image=cv2.bilateralFilter(image,9,30,20)
            #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #image=image[:,:,np.newaxis]
            images.append(image)
            measurement=float(line[3])#steering angle picked
            measurements.append(measurement)
            image_flipped = np.fliplr(image)
            images.append(image_flipped)
            measurement_flipped = -measurement
            measurements.append(measurement_flipped)
            
        if abs(float(line[3]))>0.5:
            
            image=cv2.imread(line[0])#center image picked
            image=cv2.bilateralFilter(image,9,50,50)
            #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #image=image[:,:,np.newaxis]
            images.append(image)
            measurement=float(line[3])#steering angle picked
            measurements.append(measurement)
            image_flipped = np.fliplr(image)
            images.append(image_flipped)
            measurement_flipped = -measurement
            measurements.append(measurement_flipped)
            
            image=cv2.imread(line[0])#center image picked
            #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #image=image[:,:,np.newaxis]
            measurement=float(line[3])#steering angle picked
            image,measurement=myXYtranslate(image,measurement,20)
            images.append(image)
            measurements.append(measurement)
            image_flipped = np.fliplr(image)
            images.append(image_flipped)
            measurement_flipped = -measurement
            measurements.append(measurement_flipped)
            
            
            
            image=cv2.imread(line[0])#center image picked
            #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #image=image[:,:,np.newaxis]
            measurement=float(line[3])#steering angle picked
            image,measurement=myXYtranslate(image,measurement,40)
            images.append(image)
            measurements.append(measurement)
            image_flipped = np.fliplr(image)
            images.append(image_flipped)
            measurement_flipped = -measurement
            measurements.append(measurement_flipped)
            
            
            image=cv2.imread(line[0])#center image picked
            #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #image=image[:,:,np.newaxis]
            measurement=float(line[3])#steering angle picked
            image,measurement=myXYtranslate(image,measurement,60)
            images.append(image)
            measurements.append(measurement)
            image_flipped = np.fliplr(image)
            images.append(image_flipped)
            measurement_flipped = -measurement
            measurements.append(measurement_flipped)
        
X_train=np.array(images)
y_train=np.array(measurements)

plt.hist(y_train,50)


model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=20,verbose=1)

model.save('model.h5')
print('model saved.')