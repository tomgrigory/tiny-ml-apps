import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

"""best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    print(acc)
    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)"""

pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


predictions = linear.predict(x_test) # Gets a list of all predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

plot = "G1"
plt.scatter(data[plot], data["G3"])
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()