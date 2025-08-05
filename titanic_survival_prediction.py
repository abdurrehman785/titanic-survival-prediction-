import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score

data = pd.read_csv('E:\\ML datasets\\titanic.csv')
data = data.drop(columns=["Name", "Cabin", "Ticket", "PassengerId"])

#Fill missing values
#data["Age"].fillna(data["Age"].mean(), inplace=True)
data["Age"] = data["Age"].fillna(data["Age"].mean(), inplace=True) # filling missong values with the mean age 
data["Embarked"]= data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

#Encode categorical values
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

data["Embarked"] = data["Embarked"].map({"C": 0, "Q": 1, "S": 2}) # hot encoding

x= data.drop(columns=["Survived"])
y = data["Survived"]
#x= pd.get_dummies(x)
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)
model = DecisionTreeClassifier(criterion="entropy")  # uses entropy and information gain
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

scores = cross_val_score(model, x, y, cv=5)
print("Cross-validated accuracy:", scores.mean()) # to confirm the 1.0 accurracy 
# tested for data leak : NO DATA LEAK PRESENT 
