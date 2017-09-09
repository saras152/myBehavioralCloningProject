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

os.chdir("E:\\RaNa_E\\ocv_work\\Project3\\09Sept2017")

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
    if abs(float(line[3]))>0.03:
        
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
        
            
        image=cv2.imread(line[1])#left image picked
        #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #image=image[:,:,np.newaxis]
        images.append(image)
        measurement=float(line[3])+0.2#steering angle picked
        measurements.append(measurement)
        image_flipped = np.fliplr(image)
        images.append(image_flipped)
        measurement_flipped = -measurement
        measurements.append(measurement_flipped)
        
        image=cv2.imread(line[2])#right image picked
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
            image=cv2.imread(line[0])#center image picked
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

