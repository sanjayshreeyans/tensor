# Imports
import tensorflow
import keras
import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=';')
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
best = 0
for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear= linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc = linear.score(x_test, y_test)

    if acc > best:
        print(acc, "DD")
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

            print(best, "END")


pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
print("Cofficients", linear.coef_)
print("Intercept", linear.intercept_)
d =linear.score(x_test, y_test)
print(d, "D")


predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])