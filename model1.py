# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 09:22:22 2017

@author: raghunath
"""

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda, Cropping2D,Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D


print('building the model')
model=Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5 , input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((80,20),(0,0))))

#model.add(Convolution2D(25,3,3,activation='relu',border_mode='same'))
#model.add(MaxPooling2D())
#model.add(Dropout(0.8))
#model.add(Convolution2D(35,3,3,activation='relu',border_mode='same'))
#model.add(MaxPooling2D())
#model.add(Dropout(0.8))
#model.add(Convolution2D(50,3,3,activation='relu',border_mode='same'))
#model.add(MaxPooling2D())#1
#model.add(Dropout(0.8))
#model.add(Convolution2D(12,3,3,activation='relu',border_mode='same'))
#model.add(MaxPooling2D())#2
#model.add(Dropout(0.8))
#model.add(Convolution2D(12,3,3,activation='relu',border_mode='same'))
#model.add(MaxPooling2D())#2
#model.add(Dropout(0.8))
#
#model.add(Convolution2D(12,3,3,activation='relu',border_mode='same'))
#
#model.add(Dropout(0.8))
#
#model.add(Convolution2D(12,3,3,activation='relu',border_mode='same'))
#
#model.add(Dropout(0.8))
#
#
##
##model.add(Convolution2D(128,3,3,activation='relu',border_mode='same'))
###model.add(MaxPooling2D())#3
##model.add(Dropout(0.8))
#model.add(Flatten())
##model.add(Dense(500))
#model.add(Dense(100))
#model.add(Dense(50))
#model.add(Dense(10))
#
##model.add(Convolution2D(6,3,3,activation='relu'))
##model.add(MaxPooling2D())
##model.add(Dropout(0.5))
##model.add(Convolution2D(6,5,5,activation='relu'))
##model.add(MaxPooling2D())
###model.add(Dropout(0.5))
##model.add(Flatten())
##model.add(Dense(120))
##model.add(Dense(84))
#
#model.add(Dense(1))
#
#model.summary()
#
#model.compile(loss='mse',optimizer='adam')

   # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride 
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='same',activation='relu')) 

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='same',activation='relu')) 

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='same',activation='relu')) 
#, W_regularizer=l2(0.001)

model.add(Convolution2D(64, 3, 3, border_mode='same',activation='relu')) 

model.add(Convolution2D(64, 3, 3, border_mode='same',activation='relu')) 


model.add(Flatten()) 
model.add(Dense(100,activation='relu')) 

model.add(Dense(50,activation='relu')) 

model.add(Dense(10,activation='relu')) 

model.add(Dense(1)) 
model.compile(optimizer='adam', loss='mse') 

model.summary()