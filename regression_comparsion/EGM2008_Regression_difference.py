# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:22:25 2021
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
y = datas.iloc[:, 3:4] # EGM2008 interpolated difference
x = datas.iloc[:, 2:3] # Number of the points

# NumPy Array Translate
X = x.values
Y = y.values

    
# Prints standard deviation, variance and mean of EGM2008 height anomalies
print("\nStandard Deviation of EGM2008 height anomalies: {} m".format(np.std(Y)))
print("Variance of EGM2008 height anomalies: {} m".format(np.var(Y)))
print("Mean of the EGM2008 height anomalies: {} m".format(np.mean(Y)))

# Indicating size of the scatter points
size = 5

# %%

# Linear Regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
# Plotting Linear Regression
plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")
plt.scatter(X, Y,color='blue', s=size)
plt.plot(x,lin_reg.predict(X), color = 'orange')
plt.title('Linear Regression EGM2008')
plt.xlabel('Number of the Values')
plt.ylabel('Geoid Height Anomaly as meter')
plt.show()

# Plotting Linear Regression with standard deviation lines
y1 = lin_reg.predict(X) - np.std(Y)
y2 = lin_reg.predict(X) + np.std(Y)

plt.figure(figsize=(16, 8))
plt.plot(x, y1, color="green")
plt.plot(x, y2, color="red")

plt.scatter(X, Y,color='blue', s=size)
plt.plot(x,lin_reg.predict(X), color = 'orange')
plt.title('Linear Regression EGM2008 with Standard Deviation Lines')
plt.xlabel('Number of the Values')
plt.ylabel('Geoid Height Anomaly as meter')
sns.set_style("whitegrid")
plt.legend(["\u03C3 - Predicted Value", "\u03C3 + Predicted Value", 
            "Regression line", "Height Anomalies"])
plt.show()


# Mean Squared Error
from sklearn.metrics import mean_squared_error
print("\nLinear regression mean squared error: ", mean_squared_error(y, lin_reg.predict(X)))

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
plt.title('2nd Order Polynomial Regression EGM2008')
plt.xlabel('Number of the Values')
plt.ylabel('Geoid Height Anomaly as meter')
plt.show()

# Plotting Polynomial Regression(2nd Order) with Standard Deviation Lines
y1 = lin_reg2.predict(poly_reg2.fit_transform(X)) - np.std(Y)
y2 = lin_reg2.predict(poly_reg2.fit_transform(X)) + np.std(Y)

plt.figure(figsize=(16, 8))
plt.plot(x, y1, color="green")
plt.plot(x, y2, color="red")

plt.scatter(X, Y,color='blue', s=size)
plt.plot(x,lin_reg2.predict(poly_reg2.fit_transform(X)), color = 'orange')
plt.title('2nd Order Polynomial Regression EGM2008 with Standard Deviation Lines')
plt.xlabel('Number of the Values')
plt.ylabel('Geoid Height Anomaly as meter')
sns.set_style("whitegrid")
plt.legend(["\u03C3 - Predicted Value", "\u03C3 + Predicted Value", 
            "Regression line", "Height Anomalies"])
plt.show()

# Mean Squared Error
print("\n2nd degree polynomial regression mean squared error: ", mean_squared_error(y, lin_reg2.predict(poly_reg2.fit_transform(X))))

# 4th Order Polynomial
poly_reg4 = PolynomialFeatures(degree = 4)
x_poly4 = poly_reg4.fit_transform(X)
lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly4,y)
# Plotting 4th Order Polynomial
plt.figure(figsize=(16, 8))
plt.scatter(X, Y, color = 'red', s=size)
plt.plot(X,lin_reg4.predict(poly_reg4.fit_transform(X)), color='black')
plt.title('4th Order Polynomial Regression EGM2008')
plt.xlabel('Number of the Values')
plt.ylabel('Geoid Height Anomaly as meter')
plt.show()

# Plotting Polynomial Regression(4th Order) with Standard Deviation Lines
y1 = lin_reg4.predict(poly_reg4.fit_transform(X)) - np.std(Y)
y2 = lin_reg4.predict(poly_reg4.fit_transform(X)) + np.std(Y)

plt.figure(figsize=(16, 8))
plt.plot(x, y1, color="green")
plt.plot(x, y2, color="red")

plt.scatter(X, Y,color='blue', s=size)
plt.plot(x,lin_reg4.predict(poly_reg4.fit_transform(X)), color = 'orange')
plt.title('4th Order Polynomial Regression EGM2008 with Standard Deviation Lines')
plt.xlabel('Number of the Values')
plt.ylabel('Geoid Height Anomaly as meter')
sns.set_style("whitegrid")
plt.legend(["\u03C3 - Predicted Value", "\u03C3 + Predicted Value", 
            "Regression line", "Height Anomalies"])
plt.show()

# Mean Squared Error
print("\n4th degree polynomial regression mean squared error: ", mean_squared_error(y, lin_reg4.predict(poly_reg4.fit_transform(X))))


# %%

# High Anomaly Value Prediction
print('\nPrediction of the Height Anomalies:')
print("Linear Regression:")
print(lin_reg.predict([[37.119]]))

print("\nPolinomal Regression(degree=2):")
print(lin_reg2.predict(poly_reg2.fit_transform([[37.119]])))

print("\nPolinomal Regression(degree=4):")
print(lin_reg4.predict(poly_reg4.fit_transform([[37.119]])))


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
