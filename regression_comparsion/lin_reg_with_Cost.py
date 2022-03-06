# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 23:13:09 2021

@author: doguilmak

Linear Regression with Cost Funciton

"""
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import seaborn as sns


datas=pd.read_csv("data.csv", usecols=['num', 'dif96'])

x = datas.iloc[:, 0:1] 
y = datas.iloc[:, 1:2]

x=x.values
y=y.values

def lin_reg_with_Cost(x, y):
    global alpha
    
    mean_x=np.mean(x)
    mean_y=np.mean(y)

    x_xmean=[]
    y_ymean=[]
    x_xmean_sq=[]
    y_ymean_sq=[]
    multi_mean=[]

    for i in range(len(x)):
        x_xmean.append(x[i]-mean_x)
        y_ymean.append(y[i]-mean_y)
        x_xmean_sq.append((x[i]-mean_x)**2)
        y_ymean_sq.append((y[i]-mean_y)**2)
        multi_mean.append(x_xmean[i]*y_ymean[i])

    multi_mean_sum=sum(multi_mean)
    x_xmean_sq_sum=sum(x_xmean_sq)
    y_ymean_sq_sum=sum(y_ymean_sq)

    r=multi_mean_sum/math.sqrt(x_xmean_sq_sum*y_ymean_sq_sum)
    Sy=math.sqrt(y_ymean_sq_sum/((len(x))-1))
    Sx=math.sqrt(x_xmean_sq_sum/((len(x))-1))
    alpha=r*(Sy/Sx)

    a=mean_y-alpha*mean_x
    Y=a+(alpha*np.array(x))    
    
    
    def computeCost(X,y,theta):
        m=len(y)
        predictions=X.dot(theta)
        square_err=(predictions - y)**2
    
        return 1/(2*m) * np.sum(square_err)

    data_n=datas.values
    m=len(data_n[:,-1])
    X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
    y=data_n[:,1].reshape(m,1)
    
    # Simplified Cost Funciton 
    theta=np.zeros((2,1))
    print(computeCost(X, y, theta))

    plt.figure(figsize=(16, 8))
    sns.set_style("whitegrid")
    plt.scatter(x, y)
    plt.plot(x, Y,c="red")
    plt.legend(["Regression line", "Height Anomalies"])
    plt.show()

lin_reg_with_Cost(x, y)
