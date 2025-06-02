import pandas as pd
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')


train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())

train['Fare'] = train['Fare'].fillna(train['Fare'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

train.to_csv('data/processed/train_cleaned.csv', index=False)
test.to_csv('data/processed/test_cleaned.csv', index=False)



