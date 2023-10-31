from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus
# Load data
train_data = pd.read_csv('PP_train_data.csv')
train_data_2 = pd.read_csv('PP_train_data.csv')
test_data = pd.read_csv('PP_test_data.csv')

# Split input and output
x = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
x_2 = train_data_2.iloc[:, :-1]
y_2 = train_data_2.iloc[:, -1]
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
dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)

# Calculate accuracy 
# y_prediction = dtree.predict(x_test)
# print("Accuracy:", metrics.accuracy_score(y_test, y_prediction))
y_prediction = dtree.predict(x_2)
print("Accuracy:", metrics.accuracy_score(y_2, y_prediction))

# build tree
dot_data = tree.export_graphviz(dtree, out_file=None, feature_names=test_data.columns)
graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf('ta_tree.pdf')
graph.write_pdf('ta_tree_2.pdf')

# build confusion_matrix
# print(classification_report(y_test, y_prediction))
# mat = confusion_matrix(y_test, y_prediction)
print(classification_report(y_2, y_prediction))
mat = confusion_matrix(y_2, y_prediction)
sns.heatmap(mat.T, annot=True, cmap='Blues', square=True, linewidths=0.01, linecolor='grey', fmt="d")
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()
