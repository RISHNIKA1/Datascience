import pandas as pd
import numpy as np

# 1. Create Data

data = {
    "hours" :[1,2,3,4,5,6,7,8],
    "result" :[0,0,0,0,1,1,1,1]
}

evaluvation =pd.DataFrame(data)
print(evaluvation)

# split X and y

X = evaluvation[["hours"]]
y = evaluvation["result"]

# train-test-split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

# select and train the model

from sklearn.linear_model import LogisticRegression

model= LogisticRegression()
model.fit(X_train,y_train)

# predict class(0/1)

y_pred = model.predict(X_test)
print("Predicted :",y_pred)
print("Actual :",y_test.values)

#predict probability

y_prob = model.predict_proba(X_test)
print(y_prob)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

acc = accuracy_score(y_test,y_pred)
print("Accuracy",acc)
cm =confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test,y_pred))
