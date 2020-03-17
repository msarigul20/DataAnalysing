# %% Import Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.tree import DecisionTreeRegressor

# Read File
first_100 = pd.read_csv('dataProjectCE475.csv', index_col=0, nrows=100)
first_100.index = np.arange(0, len(first_100))
last_20 = pd.read_csv('dataProjectCE475.csv', index_col=0, skiprows=range(1, 101), nrows=20)

property_columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
#defining
label_X_100 = first_100[property_columns]
label_Y_100 = first_100.Y
label_X_20 = last_20[property_columns]

#Splited
X_train, X_test, y_train, y_test = train_test_split(label_X_100, label_Y_100,test_size=0.20, random_state=0)

#Starting  Decision Tree Regression
D_T_R = DecisionTreeRegressor(random_state=0, criterion="mse")
fir_dtr = D_T_R.fit(X_train, y_train)

# Calculated Mean Square Error with Cross Validation
scores_dt = np.sqrt(-cross_val_score(fir_dtr, label_X_100, label_Y_100, cv=10, scoring='neg_mean_squared_error'))
print("MSE with CrossValidation score: " + format(np.mean(scores_dt)))
# Calculated Mean Square Error without Cross Validation
print("MSE without CrossValidation score: " + format(np.sqrt(np.sum(np.square(y_test - fir_dtr.predict(X_test))) / len(X_test))))

# DecisionTreeRegressor with GridSearchCV
max_depth = np.arange(1, 21)
score = make_scorer(mean_squared_error)
grid_cv = GridSearchCV(DecisionTreeRegressor(random_state=0),
                       param_grid={'min_samples_split': range(2, 10), 'max_depth': max_depth},
                       scoring=score, cv=10, refit=True)

grid_cv.fit(X_train, y_train)
grid_cv.best_params_
rs = grid_cv.cv_results_

# MSE of GridSearchCV
print("Root Mean Square Error of best estimator: " + format(
    np.sqrt(np.sum(np.square(y_test - grid_cv.best_estimator_.predict(X_test))) / len(X_test))))

# fitting before
grid_cv.fit(label_X_100, label_Y_100)

# predictions with model of DT
y_pred_DT = grid_cv.best_estimator_.predict(label_X_20)
predictions = D_T_R.predict(X_test)
#BEFORE THE PART OF DISPLAYING
#I used reshape method because of Scientific version of Pycharm.
#I don't have Scientific version of Pycharm so that I display as a 1D array and used for output.
#printing my predictions with Desicion Tree Regression
print("#####################################")
print("My predictions in Desicion Tree Regression: ")
#reshape for appreance like as array
tempArray = np.reshape(predictions, (len(predictions), 1))
print(label_X_20)
print("Y")
print(tempArray)
print("#####################################")
