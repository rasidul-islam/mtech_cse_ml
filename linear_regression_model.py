import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv(r'/content/Financial_Coverage.csv')
print(dataset.head())
# collecting between bmi (x) and charges (y
X = dataset['bmi'].values
Y = dataset['charges'].values
# calculate mean of x & y using an inbuilt numpy method mean()
mean_x = np.mean(X)
mean_y = np.mean(Y)
# total no.of input values
m = len(X)
# using the formula to calculate m & c
numer = 0
denom = 0
for i in range(m):
  numer+= (X[i] - mean_x)*(Y[i] - mean_y)
  denom += (X[i] - mean_x) ** 2
m = numer / denom
c = mean_y - (m * mean_x)
print (f'm = {m} \nc = {c}')
# plotting values and regression line
max_x = np.max(X) + 100
min_x = np.min(Y) - 100
# calculating line values x and y
x = np.linspace (10, 100, 5)
y = c + m * x
plt.plot(x, y, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='#ef5423', label='data points')
plt.xlabel('bmi')
plt.ylabel('charges')
plt.legend()
plt.show()
# calculating R-squared value for measuring goodness of our model.
ss_t = 0 #total sum of squares
ss_r = 0 #total sum of square of residuals
for i in range(int(m)): # val_count represents the no.of input x values
  y_pred=c + m * X[i]
  ss_t += (Y[i] - mean_y) ** 2
  ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)
print(r2)
