import numpy as np
import pandas as pd

training_data = pd.read_csv('storepurchasedata.csv')

print(training_data.describe())

X = training_data.iloc[:,:-1].values
y = training_data.iloc[:,-1].values

# print(X)
# print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=0)

print("library is installed")


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print("This is the scaled data = ",X_train,y_train)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric= 'minkowski',p=2)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:,1]

print(y_pred)
print(y_prob)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


new_prediction = classifier.predict(sc.transform(np.array([[40,20000]])))

new_prediction_proba = classifier.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]
print(new_prediction_proba,new_prediction)

new_pred = classifier.predict(sc.transform(np.array([[42,50000]])))

new_pred_proba = classifier.predict_proba(sc.transform(np.array([[42,50000]])))[:,1]
print(new_pred_proba,new_pred)


# Picking the Model and Standard Scaler
# This is really important in order to save the model and reuse it on another place 

import pickle

model_file = "classifier.pickle"
pickle.dump(classifier, open(model_file,'wb'))

scaler_file = "sc.pickle"
pickle.dump(sc, open(scaler_file,'wb'))

