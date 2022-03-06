# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:47:39 2021
@author: doguilmak

https://www.w3schools.com/python/python_ml_linear_regression.asp
https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score

"""
print(__doc__)

# %%

# Importing Global Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%

# Data Preprocessing

# Uploading Datas
datas = pd.read_csv('data.csv')
datas.info() # Looking for the missing values

# Creating correlation matrix heat map
"""
Plot rectangular data as a color-encoded matrix.
https://seaborn.pydata.org/generated/seaborn.heatmap.html
"""
plt.figure(figsize = (12,6))
#sns.heatmap(datas.corr(), annot = True)
sns.pairplot(datas)
plt.show()

# DataFrame Slice
y = datas.iloc[:, 1:2] # EGM96 geoid height anomalies
x = datas.iloc[:, 2:3] # Number of the points

# NumPy Array Translate
X = x.values
Y = y.values

# Prints standard deviation, variance and mean of EGM2008 height anomalies
print("\nStandard Deviation of EGM96 height anomalies: {} m".format(np.std(Y)))
print("Variance of EGM96 height anomalies: {} m".format(np.var(Y)))
print("Mean of the EGM96 height anomalies: {} m".format(np.mean(Y)))

# Indicating size of the scatter points
size = 5

# %%

# Linear Regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
# Plotting Linear Regression
plt.figure(figsize=(16, 8))
plt.scatter(X, Y,color='blue', s=size)
plt.plot(x,lin_reg.predict(X), color = 'orange')
plt.title('Linear Regression EGM96')
plt.xlabel('Number of the Values')
plt.ylabel('Geoid Height Anomaly as meter')
sns.set_style("whitegrid")
plt.show()

# %%

# Polynomial Regression

# 2nd Order Polynomial
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree = 2)
x_poly2 = poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2, y)
# Plotting 2nd Order Polynomial
plt.figure(figsize=(16, 8))
plt.scatter(X, Y, color = 'red', s=size)
plt.plot(X,lin_reg2.predict(poly_reg2.fit_transform(X)), color = 'green')
plt.title('2nd Order Polynomial Regression EGM96')
plt.xlabel('Number of the Values')
plt.ylabel('Geoid Height Anomaly as meter')
plt.show()

# 4th Order Polynomial
poly_reg4 = PolynomialFeatures(degree = 4)
x_poly4 = poly_reg4.fit_transform(X)
lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly4,y)
# Plotting 4th Order Polynomial
plt.figure(figsize=(16, 8)) 
plt.scatter(X, Y, color = 'red', s=size)
plt.plot(X,lin_reg4.predict(poly_reg4.fit_transform(X)), color='black')
plt.title('4th Order Polynomial Regression EGM96')
plt.xlabel('Number of the Values')
plt.ylabel('Geoid Height Anomaly as meter')
plt.show()

# %%

# Measuring Data
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1, 1)))


# Support Vector Prediction
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')  # Radial Bases Function
svr_reg.fit(x_olcekli, y_olcekli)  # Predict equivalent for each x scale value.

plt.figure(figsize=(16, 8))
plt.scatter(x_olcekli, y_olcekli, color='black', s=size)
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color='cyan')
plt.title("Support Vector Prediction EGM96")
plt.xlabel('Number of the Values')
plt.ylabel('Geoid Height Anomaly as meter')
plt.show()

# %%

# Random Forest
from sklearn.ensemble import RandomForestRegressor
# 10 Different Decision Trees
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X, Y.ravel())  # Learning Y from X

plt.figure(figsize=(16, 8))
plt.scatter(X, Y.ravel(), s=size)
plt.xlabel('Number of the Values')
plt.ylabel('Geoid Height Anomaly as meter')
plt.title("Random Forest Regressor Prediction Values EGM96")
plt.show()

# %%

# Desicion Tree
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)  # Desicion Tree Regressor
r_dt.fit(X, Y)

Z = X + 0.5
K = X - 0.4

plt.figure(figsize=(16, 8))
plt.scatter(X, Y, color='orange', s=size)
plt.plot(x, r_dt.predict(X), color='blue', linestyle='dashed')
plt.plot(x, r_dt.predict(Z), color='green', linestyle='dashed')
plt.plot(x, r_dt.predict(K), color='yellow', linestyle='dashed')
plt.title('Desicion Tree EGM96')
plt.xlabel('Number of the Values')
plt.ylabel('Geoid Height Anomaly as meter')
plt.show()

# %%

# High Anomaly Value Prediction
print('\nPrediction of the Height Anomalies:')
print("Linear Regression:")
print(lin_reg.predict([[37.119]]))

print("\nPolinomal Regression(degree=2):")
print(lin_reg2.predict(poly_reg2.fit_transform([[37.119]])))

print("\nPolinomal Regression(degree=4):")
print(lin_reg4.predict(poly_reg4.fit_transform([[37.119]])))

print("\nDecision Tree:")
print(r_dt.predict([[37.119]]))

print("\nRVS Radial Bases Function:")
print(svr_reg.predict([[37.119]]))

print("\nRandom Forest:")
print(rf_reg.predict([[37.119]]))

# %%

# R² Values of the Regressions
"""

What is R²?
The r2_score function computes the coefficient of determination, usually 
denoted as R².

It represents the proportion of variance (of y) that has been explained by 
the independent variables in the model. It provides an indication of goodness 
of fit and therefore a measure of how well unseen samples are likely to be 
predicted by the model, through the proportion of explained variance.

As such variance is dataset dependent, R² may not be meaningfully comparable 
across different datasets. Best possible score is 1.0 and it can be negative 
(because the model can be arbitrarily worse). A constant model that always 
predicts the expected value of y, disregarding the input features, would get a 
R² score of 0.0.

https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score

"""
from sklearn.metrics import r2_score

print('\n\nR² Values of the Regressions:\n')
print('Linear Regression R² value')
print(r2_score(Y, lin_reg.predict(X)))

print('\nPolynomial Regression R² value(degree=2)')
print(r2_score(Y, lin_reg2.predict(poly_reg2.fit_transform(X))))

print('\nPolynomial Regression R² value(degree=4)')
print(r2_score(Y, lin_reg4.predict(poly_reg4.fit_transform(X))))
