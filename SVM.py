import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn import svm

# Load data
train_data = pd.read_csv('PP_train_data.csv')
test_data = pd.read_csv('PP_test_data.csv')

# Split input and output
x = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
print(f"Columns of input: {list(x.columns)}")
print(f"Columns of output: {y.name}")
print(f"Shape of input: {x.shape}")
print(f"Shape of output: {y.shape}")

# Split train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(f"Shape of x_train: {x_train.shape}")
print(f"Shape of x_test: {x_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# fit
model_SVM = svm.SVC(kernel='linear',C=1,gamma='auto')
model_SVM.fit(x_train,y_train)

# Calculate accuracy
predict_SVM = model_SVM.predict(x_test)

# build confusion_matrix
print("Accuracy_SVM:", metrics.accuracy_score(y_test, predict_SVM))
print(classification_report(y_test, predict_SVM))
mat = confusion_matrix(y_test, predict_SVM)
sns.heatmap(mat.T, annot=True, cmap='Blues', square=True, linewidths=0.01, linecolor='grey', fmt="d")
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()