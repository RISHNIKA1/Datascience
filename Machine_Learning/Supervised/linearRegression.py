import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    "hours" : [1,2,3,4,5,6,7,8],
    "marks" : [35,40,50,60,65,70,75,85]
}

marks_table =pd.DataFrame(data)
print(marks_table)

# Visualization --> find a relationship

plt.scatter(marks_table["hours"],marks_table["marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

#train - test - split

from sklearn.model_selection import train_test_split

X = marks_table[["hours"]]
y = marks_table["marks"]

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

# Train Model

from sklearn.linear_model import LinearRegression

model =LinearRegression()
model.fit(X_train,y_train)

#prediction

y_prediction =model.predict(X_test)
print(y_prediction)

# Accuracy Check

from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(y_test,y_prediction)
print(error)
