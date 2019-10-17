# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:37:57 2019

@author: atul
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

#Importing the Dataset 
dataset=pd.read_excel("Data_Train.xlsx")

#Dropping the Unnecessary Column(s)
dataset=dataset.drop(['Dep_Time'],axis=1)
dataset=dataset.drop(['Arrival_Time'],axis=1)
dataset=dataset.drop(['Duration'],axis=1)
dataset=dataset.drop(['Additional_Info'],axis=1)

#Deleting the Row With Missing Values 
dataset.drop(dataset.index[9039],axis=0,inplace=True)

#Resolving the Date Column 
import datetime
for i in range(0,len(dataset['Date_of_Journey'])+1):
    if i == 9039:
        continue
    dataset.loc[i,'Date_of_Journey']=datetime.datetime.strptime(dataset.loc[i,"Date_of_Journey"], "%d/%m/%Y").strftime("%Y-%m-%d")
    dataset.loc[i,'Date_of_Journey']=datetime.datetime.strptime(dataset.loc[i,'Date_of_Journey'],"%Y-%m-%d").strftime("%A")
    if (dataset.loc[i,'Date_of_Journey']) == "Saturday" or (dataset.loc[i,'Date_of_Journey']) == "Sunday":
        dataset.loc[i,'Date_of_Journey']="Weekend"
    else:
        dataset.loc[i,'Date_of_Journey']="Weekday"
   
dataset=pd.concat([dataset,pd.get_dummies(dataset['Date_of_Journey'],prefix="is")],axis=1)
dataset.drop(['Date_of_Journey'],axis=1,inplace=True)

#Resolving Categorical Variables in Airlines Column
from sklearn import preprocessing 
le1=preprocessing.LabelEncoder()
dataset['Airline']=le1.fit_transform(dataset['Airline'])


#Resolving Catergorical Variables in Souce and Destination Column 
from sklearn import preprocessing 
le2=preprocessing.LabelEncoder()
dataset['Source']=le2.fit_transform(dataset['Source'])
dataset['Destination']=le2.fit_transform(dataset['Destination'])


#Resolving Variables in Total Stops Column 
olddat=["non-stop","1 stop","2 stops","3 stops","4 stops"]
newdat=[0,1,2,3,4]
for search,replace in zip(olddat,newdat):
    dataset['Total_Stops']=dataset['Total_Stops'].replace(search,replace)
    
#Checking the Corealtion Between Number of Stops and Price 
plt.scatter(dataset['Total_Stops'],dataset['Price'])
plt.show()

#Removing Source , Destination and '->' from Route Column 
for i in range(0,len(dataset['Route'])+1):
    if i==9039:
        continue
    x=dataset.loc[i,'Route']
    x=x[3:-3]
    x=x[3:-3]
    dataset.loc[i,'Route']=x

for i in range(0,len(dataset['Route'])+1):
    if i==9039:
        continue
    if dataset.loc[i,'Route']=='':
        dataset.loc[i,'Route']=str('NaN')
    
#Splitting the Route Column into Multiple Columns and Dropiing the Original Route Column
new = dataset['Route'].str.split("→", n = 4, expand = True) 
dataset['STOP_1']= new[0] 
dataset['STOP_2']= new[1] 
dataset['STOP_3']= new[2]
dataset['STOP_4']= new[3]  
dataset=dataset.drop(['Route'],axis=1)

#Converting the None variable to 'NaN' to encode the variable
for i in range(0,len(dataset['STOP_2'])+1):
    if i==9039:
        continue
    if dataset.loc[i,'STOP_2']==None:
        dataset.loc[i,'STOP_2']=str('Nan')
        
for i in range(0,len(dataset['STOP_3'])+1):
    if i==9039:
        continue
    if dataset.loc[i,'STOP_3']==None:
        dataset.loc[i,'STOP_3']=str('Nan')
        
for i in range(0,len(dataset['STOP_4'])+1):
    if i==9039:
        continue
    if dataset.loc[i,'STOP_4']==None:
        dataset.loc[i,'STOP_4']=str('Nan')
        
#Encoding the Categorical Variables in the various Stops Columns
from sklearn import preprocessing 
le3=preprocessing.LabelEncoder()
dataset['STOP_1']=le3.fit_transform(dataset['STOP_1'])
dataset['STOP_2']=le3.fit_transform(dataset['STOP_2'])
dataset['STOP_3']=le3.fit_transform(dataset['STOP_3'])
dataset['STOP_4']=le3.fit_transform(dataset['STOP_4'])

#Implementing Random Forest Algorithm 
X=dataset
Y=dataset['Price']
X=X.drop(['Price'],axis=1)

from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2,random_state=0) 

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=50,random_state=0)
regressor.fit(X_Train,Y_Train)
y_pred=regressor.predict(X_Test)

#Plotting the Results
plt.scatter(Y_Test,y_pred)
plt.show()

#Calculating Accuracy
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(Y_Test, y_pred))

#Calculating Accuracy
from sklearn.metrics import r2_score
r2_score(Y_Test,y_pred,multioutput='uniform_average')