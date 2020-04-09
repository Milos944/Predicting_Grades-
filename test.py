import tensorflow
from tensorflow import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot
from matplotlib import style
from sklearn.utils import shuffle

# Import our data:
data = pd.read_csv("student-mat.csv", sep=";")

# Import only attributes of our data:
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"


# Set up two arrays, one is going to define attributes and the other one will define labels:
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

# Split into four variables:
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# Next step is to implement an algorithm and use it to predict and train our data set:
# "linear" will be our model where we are going to implement our algorithm'''
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

# "acc" stands for accuracy in this case:
acc = linear.score(x_test, y_test)
print(acc)
print('Coefficient: \n', linear.coef_)
print('Intercept:   \n', linear.intercept_)

# Next is to use this data to predict grades on a real student in a real time:
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Next step is to visualise our data via style:

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])

# Next step is to set labels for our axies and print graph:
pyplot.xlabel("p")
pyplot.ylabel("Final Grade")
pyplot.show()

















