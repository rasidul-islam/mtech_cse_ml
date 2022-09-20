import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

dataset = pd.read_csv('/content/bill_authentication.csv')
X = dataset.drop('Class', axis=1)
y = dataset['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))
print("Accuracy : ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
