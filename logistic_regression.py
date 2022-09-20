import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
dataset = pd.read_csv('/content/Financial_Coverage.csv')
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
xtest = sc_x.transform(x_test)
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix : \n", cm)
print("Accuracy : ", accuracy_score(y_test, y_pred))
