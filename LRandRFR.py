import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


#read file which has data.
data = pd.read_csv("dataProjectCE475.csv")
dataFrame = pd.DataFrame(data)
#printing for check data between 99 to end of data.
print("My File :")
print(data[99:len(data)])

#mark the nan cell with Y
x_and_y_labels= dataFrame[pd.isnull(dataFrame["Y"]) == False]
x_and_y_mypart =dataFrame[pd.isnull(dataFrame["Y"])]
# our labels from x1 to x6
variables=["x1","x2","x3","x4","x5","x6"]

y2=pd.get_dummies(x_and_y_mypart["Y"])
x_and_y_mypart=x_and_y_mypart[variables]
x_and_y_mypart=pd.concat([x_and_y_mypart, y2], axis=1)
print("********************************")
print("My labels are without Y label :")
#it will prints with label Y continue parts.
print(x_and_y_mypart)
print("********************************")
#prepared random forest regressor
RFR=RandomForestRegressor()
RFR.fit(x_and_y_labels[variables], x_and_y_labels["Y"])
tempGroup=x_and_y_mypart
#doing prediction
takenRFR=RFR.predict(X=x_and_y_mypart[variables])
#to int
x_and_y_mypart["Y"]=takenRFR.astype(int)
print("/////////////////////////////////")
print(" My table of RANDOM FOREST predictions values : ")
#its here added my predicts on label Y
print(x_and_y_mypart)
realityOfRandomFR=RFR.score(x_and_y_mypart[variables], x_and_y_mypart["Y"])
print("Score of random forest which given  from Library RandomForestRegressor : ", realityOfRandomFR)
print("/////////////////////////////////")
#Linear Regression
#Prepared Linear Regression
LR=LinearRegression()
LR.fit(x_and_y_labels[variables], x_and_y_labels["Y"])
#doing prediction
takenLR=LR.predict(tempGroup[variables])
#to int
tempGroup["Y"]=takenLR.astype(int)
print("-------------------------------------")
#its here my predictions on label Y
print(" My table of LİNEAR predictions values : ")
print(tempGroup)
realityOfLinearR=LR.score(x_and_y_mypart[variables], x_and_y_mypart["Y"])
print("Score of linear which given  from Library Linear Regressor : ", realityOfLinearR)
print("---------------------------------")

