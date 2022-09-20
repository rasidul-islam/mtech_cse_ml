import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score ,classification_report

dataset = pd.read_csv('/content/Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 
pca = PCA(n_components = 2)
 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
 
explained_variance = pca.explained_variance_ratio

 
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy : ", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix : \n", confusion_matrix(y_test, y_pred))
print('\nClassification report:\n', classification_report(y_test, y_pred))
