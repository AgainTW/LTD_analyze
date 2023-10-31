import pandas as pd
# Convert non-numeric data to numerical data

train_data = pd.read_csv('train_data_2.csv')
test_data = pd.read_csv('test_data_2.csv')
NCKU_ID = pd.read_csv('NCKU_ID.csv')
ID_set = NCKU_ID['ID']

train = train_data
test = test_data

# ID
r_train, c_train = train.shape
r_test, c_test = test.shape
r_ID_set, c_ID_set = NCKU_ID.shape
for i in range(r_train):
	for j in range(r_ID_set):
		temp = str(train['ID'][i])
		if(temp[0:2]==ID_set[j]): train['ID'][i] = j
for i in range(r_test):
	for j in range(r_ID_set):
		temp = str(test['ID'][i])
		if(temp[0:2]==ID_set[j]): test['ID'][i] = j

# Gender
train['Gender'] = train['Gender'].apply(lambda x: 1 if x == 'M' else 2)
test['Gender'] = test['Gender'].apply(lambda x: 1 if x == 'M' else 2)

# Attitude
train['Attitude'] = train['Attitude'].apply(lambda x: 1 if x == 'positive' else 2)
test['Attitude'] = test['Attitude'].apply(lambda x: 1 if x == 'positive' else 2)

# Performance
train['Performance'] = train['Performance'].apply(lambda x: 1 if x == 'bad' else x)
train['Performance'] = train['Performance'].apply(lambda x: 2 if x == 'normal' else x)
train['Performance'] = train['Performance'].apply(lambda x: 3 if x == 'good' else 4)
test['Performance'] = test['Performance'].apply(lambda x: 1 if x == 'positive' else x)
test['Performance'] = test['Performance'].apply(lambda x: 2 if x == 'normal' else x)
test['Performance'] = test['Performance'].apply(lambda x: 3 if x == 'good' else 4)

# Entertainment
train['Entertainment'] = train['Entertainment'].apply(lambda x: 1 if x == 'shopping' else x)
train['Entertainment'] = train['Entertainment'].apply(lambda x: 2 if x == 'music' else x)
train['Entertainment'] = train['Entertainment'].apply(lambda x: 3 if x == 'bird-watching' else x)
train['Entertainment'] = train['Entertainment'].apply(lambda x: 4 if x == 'pokemon_go' else x)
train['Entertainment'] = train['Entertainment'].apply(lambda x: 5 if x == 'sport' else x)
train['Entertainment'] = train['Entertainment'].apply(lambda x: 6 if x == 'reading' else 7)
test['Entertainment'] = test['Entertainment'].apply(lambda x: 1 if x == 'shopping' else x)
test['Entertainment'] = test['Entertainment'].apply(lambda x: 2 if x == 'music' else x)
test['Entertainment'] = test['Entertainment'].apply(lambda x: 3 if x == 'bird-watching' else x)
test['Entertainment'] = test['Entertainment'].apply(lambda x: 4 if x == 'pokemon_go' else x)
test['Entertainment'] = test['Entertainment'].apply(lambda x: 5 if x == 'sport' else x)
test['Entertainment'] = test['Entertainment'].apply(lambda x: 6 if x == 'reading' else 7)

# Changed_Major
train['Changed_Major'] = train['Changed_Major'].apply(lambda x: 1 if x == 'TRUE' else 2)
test['Changed_Major'] = test['Changed_Major'].apply(lambda x: 1 if x == 'TRUE' else 2)

# store pre-processing
train.to_csv('PP_train_data_2.csv', index=False)
test.to_csv('PP_test_data_2.csv', index=False) 