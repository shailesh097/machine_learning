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

diabetes_x = diabetes.data[:, np.newaxis, 2]
# print(diabetes_x)

# X train data
diabetes_train_x = diabetes_x.data[:-30]
diabetes_test_x = diabetes_x.data[-30:]

# Y train data
diabetes_train_y = diabetes.target[:-30]
diabetes_test_y = diabetes.target[-30:]

#creating linear regression model
model = linear_model.LinearRegression()
model.fit(diabetes_train_x, diabetes_train_y)

#create a prediction model which shows mean squared error
diabetes_predicted_y = model.predict(diabetes_test_x)
print("Mean squared error is: ", mean_squared_error(diabetes_test_y, diabetes_predicted_y))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

plt.scatter(diabetes_test_x, diabetes_test_y)
plt.show()