import pandas as pd
import random
from sklearn.utils import shuffle

# Classification Problem : Who is good teaching assistant
'''
feature information :   (feature number are 15)
    ID : student id (one english letter + 8 number integer)
    Age : 22-32
    Gender : M, F
    Height : 140-190
    Weight : 35-120
    Birthplace : 1-22 (counties in Taiwan)
    Attitude : positive, negative
    Performance(Class performance) : great, good, normal, bad
    Sleeping(Average sleeping time) : 4-15
    Research_time : 4-10
    Entertainment : shopping, music, bird-watching, pokemon_go, sport, reading, cooking
    Cost(Cost per day) : 50, 100, 150, 200, 250, 300
    Freq_Sport(Freq of sports in one week) : 1, 2, 3, 4
    Changed_Major : 0(No), 1(Yes)
    Health : 1-5(bad to good)
    Label : 0(bad), 1(good)
absolute right rule : 
    f(x) = 2*(Age%3-1) + (Attitude==positive) - (Attitude==negative) + 0.5*(Research_time-5) + (Entertainment==pokemon_go) + 
           (Freq_Sport-2.5)^2 + (Health-2)/5 - 2*(Changed_Major==1) - (Birthplace%7==1)/(Cost/200) + (ID==Q) - 2*(ID==K2)
    if f(x)>=0,lable is 1
    else if f(x)<0,lable is 0
'''
df = pd.read_csv('NCKU_ID.csv')
ID_list = df['ID'].values.tolist()

def ID_gen(ID_list):
    ID = str(random.choice(ID_list)) + str(random.randint(0, 9)) + str(random.randint(100, 111)) + str(random.randint(0, 9)) + str(random.randint(0, 9)) + str(random.randint(0, 9))
    return ID

def G_label(temp):
    f = 2*(temp[1]%3-1) + (temp[6]=='positive') - (temp[6]=='negative') + 0.5*(temp[9]-5) + (temp[10]=='pokemon_go') + (temp[12]-2.5)**2 + (temp[14]-2)/5 - 2*(temp[13]==1) - (temp[5]%7==1)/(temp[11]/200) + (temp[0][0]=='Q') - 2*(temp[0][0:1]=='K2')
    if(f>=0):    return 1
    else: return 0


feature = ['ID', 'Age', 'Gender', 'Height', 'Weight', 'Birthplace',
           'Attitude', 'Performance', 'Sleeping', 'Research time',
           'Entertainment', 'Cost', 'Freq_Sport', 'Changed_Major',
           'Health', 'Label']
data = pd.DataFrame(columns=feature)


# data produce
for i in range(3000):
    temp = [ID_gen(ID_list), random.randint(22, 32), random.choice(['M', 'F']), random.randint(140, 190),
            random.randint(35, 120), random.randint(1,22),random.choice(['positive', 'negative']),
            random.choice(['great', 'good', 'normal', 'bad']), random.randint(4, 15), random.randint(4, 10),
            random.choice(['shopping, music', 'bird-watching', 'pokemon_go', 'sport', 'reading', 'cooking']),
            random.choice([50, 100, 150, 200, 250, 300]), random.randint(1, 4), (random.random()<0.2),
            random.randint(1, 5)]
    temp.append(G_label(temp))
    data.loc[i] = temp

data_shuffled = shuffle(data)

print(data_shuffled)
print(data_shuffled['Label'].sum())
data_shuffled.to_csv('original_data.csv', index=False)

# train data and test data split
original_data = pd.read_csv('original_data.csv')
train_data = original_data[1:2400]
test_data = original_data[2400:3000]
test_data = test_data.drop('Label', axis = 1)
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print(train_data['Label'].sum())
