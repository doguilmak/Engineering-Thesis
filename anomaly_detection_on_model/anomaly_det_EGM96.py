# -*- coding: utf-8 -*-
"""
Created on Wed May 12 19:39:03 2021

@author: doguilmak

"""

# %%

# Importing Global Libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib import patches
import seaborn as sns


# %%

# Data Preprocessing

# Uploading Datas
df= pd.read_csv('data.csv', sep=",", decimal='.')
print(df.head())

df = df[['EGM96', 'dif96']]

# Looking for the missing values
df.info()
df = df.dropna()
df = df.to_numpy()

# %%

# Covariance matrix
covariance = np.cov(df, rowvar=False)

# Covariance matrix power of -1
covariance_pm1 = np.linalg.matrix_power(covariance, -1)

# Center point
centerpoint = np.mean(df , axis=0)

# %%

# Distances between center point and 
distances = []
for i, val in enumerate(df):
      p1 = val
      p2 = centerpoint
      distance = (p1-p2).T.dot(covariance_pm1).dot(p1-p2)
      distances.append(distance)
distances = np.array(distances)


# %%

# Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers 
cutoff = chi2.ppf(0.95, df.shape[1])

# %%

# Index of outliers

outlierIndexes = np.where(distances > cutoff)

print('Founded Outlier Observations')
print(df[distances > cutoff, :])
outliner_obs = df[distances > cutoff, :]

# %%

## Finding ellipse dimensions 
pearson = covariance[0, 1]/np.sqrt(covariance[0, 0] * covariance[1, 1])
#ell_radius_x = np.sqrt(1 + pearson)
#ell_radius_y = np.sqrt(1 - pearson)
lambda_, v = np.linalg.eig(covariance)
lambda_ = np.sqrt(lambda_)

# %%

# Ellipse patch
ellipse = patches.Ellipse(
    xy=(centerpoint[0], centerpoint[1]),
    width=lambda_[0]*np.sqrt(cutoff)*2, 
    height=lambda_[1]*np.sqrt(cutoff)*2,
    angle=np.rad2deg(np.arccos(v[0, 0])), edgecolor='#fab1a0')

ellipse.set_facecolor('#0984e3')
ellipse.set_alpha(0.4)

# %%

sns.set_style("whitegrid")
fig = plt.figure(figsize = (16, 8))
ax = plt.subplot()
ax.add_artist(ellipse)
plt.scatter(df[:, 0], df[:, 1], marker="x")
plt.scatter(outliner_obs[:, :1], outliner_obs[:, 1:2], c='red')
plt.legend(['Non-Outlier Values According to Mahalanobis Distance','Outlier values according to Mahalanobis Distance'])
plt.title("Founding Outlier Observations from EGM96 Datas Using Mahalanobis Distance", fontsize=15)
plt.xlabel("Longitude")
plt.ylabel("Difference between ellipsoidal heights and normal heights with linear interpolation(m)", fontsize=8)
plt.savefig('plot_EGM96.png', dpi=300, bbox_inches='tight')  # Saving plot as .png
plt.show()
