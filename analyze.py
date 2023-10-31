import pandas as pd

# Load data
train_data = pd.read_csv('PP_train_data_2.csv')

# Calculate the probability of different label
a = train_data['Label'].value_counts()
print(" label 1 :", a[1]/len(train_data))
print(" label 0 :", a[0]/len(train_data))
