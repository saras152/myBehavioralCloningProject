# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 09:23:15 2017

@author: raghunath
"""

model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=20,verbose=1)

model.save('model.h5')
print('model saved.')