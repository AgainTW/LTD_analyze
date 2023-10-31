import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)

# Calculate accuracy
y_prediction = knn.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_prediction))

# build confusion_matrix
print(classification_report(y_test, y_prediction))
mat = confusion_matrix(y_test, y_prediction)
sns.heatmap(mat.T, annot=True, cmap='Blues', square=True, linewidths=0.01, linecolor='grey', fmt="d")
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()
