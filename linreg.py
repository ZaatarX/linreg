import matplotlib.pyplot as pyplot
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style

# data preparation

unfiltered_data = pd.read_csv("student-mat.csv", sep=";")
filtered_data = unfiltered_data[["G1", "G2",
                                 "G3", "studytime", "failures", "absences"]]

wanted_label = "G3"

feature = np.array(filtered_data.drop([wanted_label], 1))
label = np.array(filtered_data[wanted_label])

train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(
    feature, label, test_size=0.1)

# function to train and save the best model


def fit_n_save():
    best_fit = 0.0

    for _ in range(5000):

        # split data

        train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(
            feature, label, test_size=0.1)

        # create and fit model

        linear = linear_model.LinearRegression()
        linear.fit(train_data, train_label)

        # check accuracy

        accuracy = linear.score(test_data, test_label)

        # saving the model

        if accuracy > best_fit:
            best_fit = accuracy
            print("New best fit model has an accuracy of: \n", best_fit)
            with open("student_model.pickle", "wb") as f:
                pickle.dump(linear, f)


# fit_n_save()

# loading pickled model

pickle_rick = open("student_model.pickle", "rb")
linear = pickle.load(pickle_rick)

# some stats

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

# show range of predictions

predictions = linear.predict(test_data)

for i in range(len(predictions)):
    print(predictions[i], test_data[i], test_label[i])

# plot

x1_axis = "G1"
x2_axis = "G2"
y_axis = "G3"
style.use("ggplot")

pyplot.scatter(
    (filtered_data[x1_axis] + filtered_data[x2_axis])/2, filtered_data[y_axis])
pyplot.xlabel("Average Grade (G1 + G2 / 2)")
pyplot.ylabel("Final Grade")
pyplot.show()
