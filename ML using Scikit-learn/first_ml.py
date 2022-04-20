"""
A machine learning model takes features as input and gives label as output.  For example, a machine learning model is given a picture of a cat. 
By looking at it's features the model has to determine if it is a cat or not. A cat has unique set of ears which the ML model already knows. 
So the model gives the ears as input which is also called a feature in ML term; 
by looking at the feature of cat's ear then the ML model lables the given picture as a cat.
"""

from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# print out all the keys for the diabetes dataset
# print(diabetes.keys())
# keys = (['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

#print out the description of the imported dataset
# print(diabetes.DESCR)

diabetes_x = diabetes.data
# print(diabetes_x)

# X train data
# taking last 350 rows as training dataset and first 92 rows as testing dataset(total 442 rows of data is available)
x_train = diabetes_x[:-350] 
x_test = diabetes_x[:92]

# Y train data
y_train = diabetes.target[:-350]
y_test = diabetes.target[:92]

#creating linear regression model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

#create a prediction model which shows mean squared error
predicted_y = model.predict(x_test)
print("Mean squared error is: ", mean_squared_error(y_test, predicted_y))

# print("Weights: ", model.coef_)
# print("Intercept: ", model.intercept_)

# plt.scatter(x_test, y_test)
# plt.plot(x_test, predicted_y)
# plt.show()