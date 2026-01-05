import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from KNNclassifier import KNN


iris =datasets.load_iris()
X,y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=42)

clf = KNN(3)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)

accuracy = np.sum(predictions == y_test)/len(y_test)
print(accuracy)