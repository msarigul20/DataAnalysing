#Importing
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, train_test_split

# Read File
first_100 = pd.read_csv('dataProjectCE475.csv', index_col=0, nrows=100)
first_100.index = np.arange(0, len(first_100))
last_20 = pd.read_csv('dataProjectCE475.csv', index_col=0, skiprows=range(1, 101), nrows=20)

property_columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
#initializing
label_X_100 = first_100[property_columns]
label_Y_100 = first_100.Y
label_X_20 = last_20[property_columns]
#Spliting
X_train, X_test, y_train, y_test = train_test_split(label_X_100, label_Y_100, test_size=0.20, random_state=0)
#starting LASSO regression
LasReg = LassoCV(cv=5, random_state=0).fit(X_train, y_train)
# Calculated Mean Square Error with Cross Validation
lasScore = np.sqrt(-cross_val_score(LasReg, label_X_100, label_Y_100, cv=10, scoring='neg_mean_squared_error'))
print("MSE with CV: " + format(np.mean(lasScore)))
# Calculated Mean Square Error without Cross Validation
print("MSE without CV: " + format(np.sqrt(np.sum(np.square(y_test - LasReg.predict(X_test))) / len(X_test))))

# fitting before
LasReg.fit(label_X_100, label_Y_100)

# lasso predicts
y_lasso_pred = LasReg.predict(label_X_20)
predictions = LasReg.predict(X_test)
#BEFORE THE PART OF DISPLAYING
#I used reshape method because of Scientific version of Pycharm.
#I don't have Scientific version of Pycharm so that I display as a 1D array and used for output.
#printing my predictions with Lasso Regression
print("#####################################")
print("My predictions in LASSO Regression: ")
#reshape for appreance like as array
tempArray = np.reshape(predictions, (len(predictions), 1))
print(tempArray)
print("#####################################")





