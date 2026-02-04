from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# load dataset

iris = load_iris()

X=iris.data
y= iris.target

df =pd.DataFrame(X,columns =iris.feature_names)
df['target']=y
df.head()

# train - test - split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Select and train the model

model = DecisionTreeClassifier(criterion="gini",random_state=42)
model.fit(X_train,y_train)

# predict

y_pred = model.predict(X_test)
print(y_pred)

# Evaluavte
#from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#Accuracy
acc = accuracy_score(y_test,y_pred)
print("Accuracy :",acc)

#Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix :",cm)

#classification Report
print("Classification Report :",classification_report(y_test,y_pred))


# Tree visualization

#from sklearn.tree import plot_tree
#import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plot_tree(model,feature_names = iris.feature_names,class_names = iris.target_names, filled=True)
plt.show()
