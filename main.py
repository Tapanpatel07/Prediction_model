import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
# from xgboost import XGBRegressor
import matplotlib.pyplot as plt

diabetes_dataset = pd.read_csv('diabetes.csv')
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
scaler = StandardScaler()
scaler.fit(X)  
standardized_data = scaler.transform(X) 
X = standardized_data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)


input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 5100)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)


if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
