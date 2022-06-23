import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

diabetes_x = np.array([[1],[2],[3]])
diabetes_y = np.array([3,2,4])

diabetes_x_train = diabetes_x
diabetes_x_test = diabetes_x

diabetes_y_train = diabetes_y
diabetes_y_test = diabetes_y

model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train) 

diabetes_y_predicted = model.predict(diabetes_x_test)
print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

plt.scatter(diabetes_x_test, diabetes_y_test)
plt.plot(diabetes_x_test, diabetes_y_predicted)
plt.show()