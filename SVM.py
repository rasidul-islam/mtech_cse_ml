import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('/content/DLBCL.csv')
X = dataset.iloc[:, :12].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #for Linear
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
ac = accuracy_score(y_test, y_pred)

print('linear: ')
print(ac)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# for Polynomial

svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
ac = accuracy_score(y_test, y_pred)

print('Polynomial: ')
print(ac)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# for RBF

svm = SVC(kernel='rbf', random_state=1, gamma=0.1, C=0.02)
svm.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
ac = accuracy_score(y_test, y_pred)

print('RBF: ')
print(ac)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)
